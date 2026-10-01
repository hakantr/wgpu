//! Pending command buffer limit of a Metal queue imported with
//! `queue_from_raw_with_pending_limit`.

use std::{
    num::NonZeroUsize,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
};

use parking_lot::Mutex;
use wgpu::hal::{api::Metal, metal};
use wgpu_test::{
    apply, gpu_test, FailureCase, GpuTestConfiguration, GpuTestInitializer, TestParameters,
};

pub fn all_tests(vec: &mut Vec<GpuTestInitializer>) {
    vec.extend([
        IMPORTED_QUEUE_PENDING_LIMIT_REFUSES_WITHOUT_DEVICE_LOSS,
        IMPORTED_QUEUE_PENDING_LIMIT_RELEASE_PATHS,
        IMPORTED_QUEUE_PENDING_LIMIT_SUBMIT_TIME_RESOLVE,
        IMPORTED_QUEUE_PENDING_LIMIT_SCOPE_RULES,
        IMPORTED_QUEUE_PENDING_LIMIT_INTERNAL_OPENS,
        IMPORTED_QUEUE_PENDING_LIMIT_NORMAL_QUEUE_UNCHANGED,
    ]);
}

fn metal_only() -> TestParameters {
    TestParameters::default().skip(FailureCase::backend(
        wgpu::Backends::all() - wgpu::Backends::METAL,
    ))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ErrorKind {
    OutOfMemory,
    Validation,
    Internal,
}

struct Imported {
    device: wgpu::Device,
    queue: wgpu::Queue,
    lost: Arc<AtomicUsize>,
    errors: Arc<Mutex<Vec<(ErrorKind, String)>>>,
    src: wgpu::Buffer,
    dst: wgpu::Buffer,
}

impl Imported {
    fn new(ctx: &wgpu_test::TestingContext, limit: usize) -> Self {
        let limits = ctx.adapter.limits();
        let mut open = {
            let hal_adapter = unsafe { ctx.adapter.as_hal::<Metal>() }.expect("metal adapter");
            unsafe {
                use wgpu::hal::Adapter as _;
                hal_adapter.open(
                    wgpu::Features::empty(),
                    &limits,
                    &wgpu::MemoryHints::default(),
                )
            }
            .expect("open")
        };
        let timestamp_period = {
            use wgpu::hal::Queue as _;
            unsafe { open.queue.get_timestamp_period() }
        };
        open.queue = unsafe {
            metal::Queue::queue_from_raw_with_pending_limit(
                open.queue.as_raw().into(),
                timestamp_period,
                NonZeroUsize::new(limit).unwrap(),
            )
        };
        let (device, queue) = unsafe {
            ctx.adapter.create_device_from_hal::<Metal>(
                open,
                &wgpu::DeviceDescriptor {
                    label: Some("imported"),
                    required_features: wgpu::Features::empty(),
                    required_limits: limits,
                    ..Default::default()
                },
            )
        }
        .expect("create_device_from_hal");

        let lost = Arc::new(AtomicUsize::new(0));
        let lost_in_callback = Arc::clone(&lost);
        device.set_device_lost_callback(move |_, _| {
            lost_in_callback.fetch_add(1, Ordering::SeqCst);
        });
        let errors = Arc::new(Mutex::new(Vec::new()));
        let errors_in_handler = Arc::clone(&errors);
        device.on_uncaptured_error(Arc::new(move |error| {
            let kind = match error {
                wgpu::Error::OutOfMemory { .. } => ErrorKind::OutOfMemory,
                wgpu::Error::Validation { .. } => ErrorKind::Validation,
                wgpu::Error::Internal { .. } => ErrorKind::Internal,
            };
            errors_in_handler.lock().push((kind, error.to_string()));
        }));

        let buffer = |usage| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: 4,
                usage,
                mapped_at_creation: false,
            })
        };
        let src = buffer(wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST);
        let dst = buffer(wgpu::BufferUsages::COPY_DST);

        let imported = Self {
            device,
            queue,
            lost,
            errors,
            src,
            dst,
        };
        // Device creation leaves the pending writes command buffer open.
        imported.queue.submit([]);
        assert_eq!(imported.pending(), 0);
        imported
    }

    fn hal<R>(&self, f: impl FnOnce(&metal::Queue) -> R) -> R {
        let queue = unsafe { self.queue.as_hal::<Metal>() }.expect("metal queue");
        f(&queue)
    }

    fn pending(&self) -> usize {
        self.hal(|queue| queue.pending_command_buffers())
    }

    fn begin_scope(&self) {
        self.hal(|queue| queue.begin_pending_limit_scope())
            .expect("begin_pending_limit_scope");
    }

    fn end_scope(&self) -> Option<metal::PendingLimitFault> {
        self.hal(|queue| queue.end_pending_limit_scope())
            .expect("end_pending_limit_scope")
    }

    fn copy(&self) -> wgpu::CommandBuffer {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encoder.copy_buffer_to_buffer(&self.src, 0, &self.dst, 0, 4);
        encoder.finish()
    }

    fn take_errors(&self) -> Vec<(ErrorKind, String)> {
        core::mem::take(&mut *self.errors.lock())
    }

    fn assert_not_lost(&self) {
        assert_eq!(self.lost.load(Ordering::SeqCst), 0, "device lost");
    }
}

fn fault(limit: usize, outstanding: usize) -> Option<metal::PendingLimitFault> {
    Some(metal::PendingLimitFault { limit, outstanding })
}

#[apply(gpu_test!)]
static IMPORTED_QUEUE_PENDING_LIMIT_REFUSES_WITHOUT_DEVICE_LOSS: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(metal_only())
        .run_sync(|ctx| {
            let imported = Imported::new(&ctx, 2);
            assert_eq!(
                imported.hal(|queue| queue.pending_limit()),
                NonZeroUsize::new(2)
            );

            imported.begin_scope();
            let first = imported.copy();
            assert_eq!(imported.pending(), 1);
            let second = imported.copy();
            assert_eq!(imported.pending(), 2);
            let refused = imported.copy();
            assert_eq!(imported.pending(), 2);
            assert_eq!(imported.end_scope(), fault(2, 2));

            let errors = imported.take_errors();
            assert!(!errors.is_empty());
            assert!(errors.iter().all(|(kind, _)| *kind == ErrorKind::Internal));
            imported.assert_not_lost();

            imported.queue.submit([refused]);
            assert_eq!(imported.pending(), 2);
            assert!(!imported.take_errors().is_empty());
            imported.assert_not_lost();

            imported.queue.submit([first]);
            assert_eq!(imported.pending(), 1);

            imported.begin_scope();
            let third = imported.copy();
            assert_eq!(imported.pending(), 2);
            assert_eq!(imported.end_scope(), None);

            imported.queue.submit([second, third]);
            assert_eq!(imported.pending(), 0);
            imported
                .device
                .poll(wgpu::PollType::wait_indefinitely())
                .unwrap();
            assert!(imported.take_errors().is_empty());
            imported.assert_not_lost();
        });

#[apply(gpu_test!)]
static IMPORTED_QUEUE_PENDING_LIMIT_RELEASE_PATHS: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(metal_only())
        .run_sync(|ctx| {
            let imported = Imported::new(&ctx, 1);

            // Dropped without submission.
            let unsubmitted = imported.copy();
            assert_eq!(imported.pending(), 1);
            drop(unsubmitted);
            assert_eq!(imported.pending(), 0);

            // Discarded after an encoding error.
            let mut encoder = imported
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            encoder.copy_buffer_to_buffer(&imported.src, 0, &imported.dst, 0, 4);
            encoder.copy_buffer_to_buffer(&imported.src, 0, &imported.dst, 0, 8);
            let invalid = encoder.finish();
            drop(invalid);
            assert_eq!(imported.pending(), 0);
            let errors = imported.take_errors();
            assert!(!errors.is_empty());
            assert!(errors
                .iter()
                .all(|(kind, _)| *kind == ErrorKind::Validation));

            // Submitted.
            let submitted = imported.copy();
            assert_eq!(imported.pending(), 1);
            imported.queue.submit([submitted]);
            assert_eq!(imported.pending(), 0);

            imported.begin_scope();
            let next = imported.copy();
            assert_eq!(imported.end_scope(), None);
            imported.queue.submit([next]);
            assert_eq!(imported.pending(), 0);
            assert!(imported.take_errors().is_empty());
            imported.assert_not_lost();
        });

#[apply(gpu_test!)]
static IMPORTED_QUEUE_PENDING_LIMIT_SUBMIT_TIME_RESOLVE: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(metal_only())
        .run_sync(|ctx| {
            let imported = Imported::new(&ctx, 2);
            let query_set = imported.device.create_query_set(&wgpu::QuerySetDescriptor {
                label: None,
                ty: wgpu::QueryType::Occlusion,
                count: 1,
            });
            let destination = imported.device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: 256,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let resolve = || {
                let mut encoder = imported
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
                encoder.resolve_query_set(&query_set, 0..1, &destination, 0);
                encoder.finish()
            };

            // `resolve_query_set` opens one command buffer when the encoder finishes, and the
            // deferred resolve opens another during the submission.
            let holding = imported.copy();
            let deferred = resolve();
            assert_eq!(imported.pending(), 2);

            imported.begin_scope();
            imported.queue.submit([deferred]);
            assert_eq!(imported.end_scope(), fault(2, 2));
            assert_eq!(imported.pending(), 1);
            let errors = imported.take_errors();
            assert!(!errors.is_empty());
            assert!(errors.iter().all(|(kind, _)| *kind == ErrorKind::Internal));
            imported.assert_not_lost();

            imported.queue.submit([holding]);
            assert_eq!(imported.pending(), 0);

            imported.begin_scope();
            imported.queue.submit([resolve()]);
            assert_eq!(imported.end_scope(), None);
            assert_eq!(imported.pending(), 0);
            assert!(imported.take_errors().is_empty());
            imported.assert_not_lost();
        });

#[apply(gpu_test!)]
static IMPORTED_QUEUE_PENDING_LIMIT_SCOPE_RULES: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(metal_only())
    .run_sync(|ctx| {
        let imported = Imported::new(&ctx, 1);

        assert_eq!(
            imported.hal(|queue| queue.end_pending_limit_scope()),
            Err(metal::PendingLimitScopeError::NotActive)
        );
        imported.begin_scope();
        assert_eq!(
            imported.hal(|queue| queue.begin_pending_limit_scope()),
            Err(metal::PendingLimitScopeError::AlreadyActive)
        );
        assert_eq!(imported.end_scope(), None);

        // A refusal outside a scope is not recorded.
        let holding = imported.copy();
        let refused = imported.copy();
        imported.begin_scope();
        assert_eq!(imported.end_scope(), None);

        // The first fault of a scope is kept.
        imported.begin_scope();
        let first_refused = imported.copy();
        imported.queue.write_buffer(&imported.src, 0, &[1, 2, 3, 4]);
        assert_eq!(imported.pending(), 2);
        let second_refused = imported.copy();
        assert_eq!(imported.end_scope(), fault(1, 1));

        // An earlier fault does not leak into the next scope.
        imported.begin_scope();
        assert_eq!(imported.end_scope(), None);

        drop((refused, first_refused, second_refused));
        imported.queue.submit([]);
        assert_eq!(imported.pending(), 1);
        drop(holding);
        assert_eq!(imported.pending(), 0);
        assert!(imported
            .take_errors()
            .iter()
            .all(|(kind, _)| *kind == ErrorKind::Internal));
        imported.assert_not_lost();
    });

#[apply(gpu_test!)]
static IMPORTED_QUEUE_PENDING_LIMIT_INTERNAL_OPENS: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(metal_only())
        .run_sync(|ctx| {
            let imported = Imported::new(&ctx, 1);

            imported.begin_scope();
            let holding = imported.copy();
            assert_eq!(imported.pending(), 1);
            imported.queue.write_buffer(&imported.src, 0, &[1, 2, 3, 4]);
            assert_eq!(imported.pending(), 2);
            imported.queue.submit([holding]);
            assert_eq!(imported.pending(), 0);
            assert_eq!(imported.end_scope(), None);

            imported
                .device
                .poll(wgpu::PollType::wait_indefinitely())
                .unwrap();
            assert!(imported.take_errors().is_empty());
            imported.assert_not_lost();
        });

#[apply(gpu_test!)]
static IMPORTED_QUEUE_PENDING_LIMIT_NORMAL_QUEUE_UNCHANGED: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(metal_only())
        .run_sync(|ctx| {
            let queue = unsafe { ctx.queue.as_hal::<Metal>() }.expect("metal queue");
            assert_eq!(queue.pending_limit(), None);
            assert_eq!(
                queue.begin_pending_limit_scope(),
                Err(metal::PendingLimitScopeError::NoPendingLimit)
            );
            assert_eq!(
                queue.end_pending_limit_scope(),
                Err(metal::PendingLimitScopeError::NoPendingLimit)
            );
            drop(queue);

            let buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: 4,
                usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let command_buffers: Vec<_> = (0..8)
                .map(|_| {
                    let mut encoder = ctx
                        .device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
                    encoder.clear_buffer(&buffer, 0, None);
                    encoder.finish()
                })
                .collect();
            ctx.queue.submit(command_buffers);
            ctx.device
                .poll(wgpu::PollType::wait_indefinitely())
                .unwrap();
        });
