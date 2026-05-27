use std::sync::Arc;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;
use web_time::Instant;
use wgpu::InstanceDescriptor;
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::ActiveEventLoop,
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowId},
};

const TRI: [([f32; 3], [f32; 3]); 3] = [
    ([1.0, -1.0, 0.0], [1.0, 0.0, 0.0]),
    ([-1.0, -1.0, 0.0], [0.0, 1.0, 0.0]),
    ([0.0, 1.0, 0.0], [0.0, 0.0, 1.0]),
];

const SHADER: &str = "
struct O { @builtin(position) p: vec4<f32>, @location(0) c: vec4<f32> }
@vertex fn vs(@location(0) p: vec4<f32>, @location(1) c: vec4<f32>) -> O { return O(p, c); }
@fragment fn fs(o: O) -> @location(0) vec4<f32> { return o.c; }
";

#[cfg(target_arch = "wasm32")]
fn canvas() -> wgpu::web_sys::HtmlCanvasElement {
    wgpu::web_sys::window()
        .unwrap()
        .document()
        .unwrap()
        .get_element_by_id("canvas")
        .unwrap()
        .dyn_into()
        .unwrap()
}

struct State {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    pipeline: wgpu::RenderPipeline,
    vbuf: wgpu::Buffer,
    egui_renderer: egui_wgpu::Renderer,
    egui_state: egui_winit::State,
    model: nalgebra_glm::Mat4,
    last: Instant,
    size: (u32, u32),
}

impl State {
    async fn new(window: Arc<Window>, w: u32, h: u32) -> Self {
        let instance =
            wgpu::Instance::new(InstanceDescriptor::new_without_display_handle_from_env());
        let surface = instance.create_surface(window.clone()).unwrap();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                compatible_surface: Some(&surface),
                ..Default::default()
            })
            .await
            .unwrap();
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                required_limits: wgpu::Limits::default().using_resolution(adapter.limits()),
                ..Default::default()
            })
            .await
            .unwrap();
        let config = surface.get_default_config(&adapter, w, h).unwrap();
        surface.configure(&device, &config);
        let format = config.format;
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let attrs = wgpu::vertex_attr_array![0 => Float32x4, 1 => Float32x4];
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: None,
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: 32,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &attrs,
                }],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs"),
                compilation_options: Default::default(),
                targets: &[Some(format.into())],
            }),
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });
        let vbuf = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 96,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let egui_renderer = egui_wgpu::Renderer::new(
            &device,
            format,
            egui_wgpu::RendererOptions {
                msaa_samples: 1,
                ..Default::default()
            },
        );
        let ctx = egui::Context::default();
        let viewport = ctx.viewport_id();
        let egui_state = egui_winit::State::new(ctx, viewport, &window, None, None, None);
        Self {
            window,
            surface,
            device,
            queue,
            config,
            pipeline,
            vbuf,
            egui_renderer,
            egui_state,
            model: nalgebra_glm::Mat4::identity(),
            last: Instant::now(),
            size: (w, h),
        }
    }

    fn resize(&mut self, w: u32, h: u32) {
        self.size = (w, h);
        self.config.width = w;
        self.config.height = h;
        self.surface.configure(&self.device, &self.config);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, event: WindowEvent) {
        if self
            .egui_state
            .on_window_event(&self.window, &event)
            .consumed
        {
            self.window.request_redraw();
            return;
        }
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(s) if s.width > 0 && s.height > 0 => {
                self.resize(s.width, s.height)
            }
            WindowEvent::KeyboardInput { event: key, .. }
                if key.physical_key == PhysicalKey::Code(KeyCode::Escape) =>
            {
                event_loop.exit()
            }
            WindowEvent::RedrawRequested => {
                self.render();
                self.window.request_redraw();
            }
            _ => {}
        }
    }

    fn render(&mut self) {
        let now = Instant::now();
        let dt = (now - self.last).as_secs_f32();
        self.last = now;

        #[cfg(target_arch = "wasm32")]
        {
            let c = canvas();
            if (c.width(), c.height()) != self.size && c.width() > 0 && c.height() > 0 {
                self.resize(c.width(), c.height());
            }
        }
        let (w, h) = self.size;

        #[cfg(not(target_arch = "wasm32"))]
        let raw = self.egui_state.take_egui_input(&self.window);
        #[cfg(target_arch = "wasm32")]
        let mut raw = self.egui_state.take_egui_input(&self.window);
        #[cfg(target_arch = "wasm32")]
        {
            let ppp = self.egui_state.egui_ctx().pixels_per_point();
            let size = egui::vec2(w as f32 / ppp, h as f32 / ppp);
            raw.screen_rect = Some(egui::Rect::from_min_size(egui::Pos2::ZERO, size));
        }

        let out = self.egui_state.egui_ctx().run_ui(raw, |ui| {
            ui.label("wgpu + egui");
        });
        self.egui_state
            .handle_platform_output(&self.window, out.platform_output);
        let jobs = self
            .egui_state
            .egui_ctx()
            .tessellate(out.shapes, out.pixels_per_point);
        let desc = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [w, h],
            pixels_per_point: out.pixels_per_point,
        };
        for (id, delta) in &out.textures_delta.set {
            self.egui_renderer
                .update_texture(&self.device, &self.queue, *id, delta);
        }
        for id in &out.textures_delta.free {
            self.egui_renderer.free_texture(id);
        }

        self.model = nalgebra_glm::rotate(
            &self.model,
            30f32.to_radians() * dt,
            &nalgebra_glm::Vec3::y(),
        );
        let proj = nalgebra_glm::perspective_lh_zo(
            w as f32 / h.max(1) as f32,
            80f32.to_radians(),
            0.1,
            1e3,
        );
        let view = nalgebra_glm::look_at_lh(
            &nalgebra_glm::vec3(0.0, 0.0, 3.0),
            &nalgebra_glm::vec3(0.0, 0.0, 0.0),
            &nalgebra_glm::Vec3::y(),
        );
        let mvp = proj * view * self.model;
        let mut data = [0f32; 24];
        for (i, (p, c)) in TRI.iter().enumerate() {
            let v = mvp * nalgebra_glm::vec4(p[0], p[1], p[2], 1.0);
            data[i * 8..i * 8 + 8].copy_from_slice(&[v.x, v.y, v.z, v.w, c[0], c[1], c[2], 1.0]);
        }
        self.queue
            .write_buffer(&self.vbuf, 0, bytemuck::cast_slice(&data));

        let mut enc = self.device.create_command_encoder(&Default::default());
        self.egui_renderer
            .update_buffers(&self.device, &self.queue, &mut enc, &jobs, &desc);
        let frame = loop {
            match self.surface.get_current_texture() {
                wgpu::CurrentSurfaceTexture::Success(f)
                | wgpu::CurrentSurfaceTexture::Suboptimal(f) => break f,
                wgpu::CurrentSurfaceTexture::Outdated => {
                    self.surface.configure(&self.device, &self.config)
                }
                other => panic!("{other:?}"),
            }
        };
        let target = frame.texture.create_view(&Default::default());
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &target,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.19,
                            g: 0.24,
                            b: 0.42,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_vertex_buffer(0, self.vbuf.slice(..));
            pass.draw(0..3, 0..1);
            self.egui_renderer
                .render(&mut pass.forget_lifetime(), &jobs, &desc);
        }
        self.queue.submit([enc.finish()]);
        frame.present();
    }
}

#[derive(Default)]
pub struct App {
    state: Option<State>,
    #[cfg(target_arch = "wasm32")]
    pending: Option<futures::channel::oneshot::Receiver<State>>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }
        #[cfg(not(target_arch = "wasm32"))]
        let attrs = Window::default_attributes();
        #[cfg(target_arch = "wasm32")]
        let (attrs, w, h) = {
            use winit::platform::web::WindowAttributesExtWebSys;
            let canvas = canvas();
            let (w, h) = (canvas.width(), canvas.height());
            (Window::default_attributes().with_canvas(Some(canvas)), w, h)
        };
        let window = Arc::new(event_loop.create_window(attrs).unwrap());
        #[cfg(not(target_arch = "wasm32"))]
        {
            let _ = env_logger::try_init();
            let s = window.inner_size();
            self.state = Some(pollster::block_on(State::new(window, s.width, s.height)));
        }
        #[cfg(target_arch = "wasm32")]
        {
            console_error_panic_hook::set_once();
            let _ = console_log::init();
            let (tx, rx) = futures::channel::oneshot::channel();
            self.pending = Some(rx);
            wasm_bindgen_futures::spawn_local(async move {
                let _ = tx.send(State::new(window, w, h).await);
            });
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        #[cfg(target_arch = "wasm32")]
        if let Some(rx) = self.pending.as_mut()
            && let Ok(Some(state)) = rx.try_recv()
        {
            self.pending = None;
            state.window.request_redraw();
            self.state = Some(state);
        }
        if let Some(state) = self.state.as_mut() {
            state.window_event(event_loop, event);
        }
    }
}
