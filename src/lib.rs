#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

use std::sync::Arc;
use web_time::{Duration, Instant};
use wgpu::{InstanceDescriptor, util::DeviceExt};
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::WindowEvent,
    window::{Theme, Window},
};

// ── Vertex / Uniform ────────────────────────────────────────────────────────

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    position: [f32; 4],
    color: [f32; 4],
}

impl Vertex {
    fn attrs() -> Vec<wgpu::VertexAttribute> {
        wgpu::vertex_attr_array![0 => Float32x4, 1 => Float32x4].to_vec()
    }
    fn layout(attrs: &[wgpu::VertexAttribute]) -> wgpu::VertexBufferLayout<'_> {
        wgpu::VertexBufferLayout {
            array_stride: size_of::<Vertex>() as _,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: attrs,
        }
    }
}

#[repr(C)]
#[derive(Default, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct UniformData {
    mvp: nalgebra_glm::Mat4,
}

// ── Uniform binding ──────────────────────────────────────────────────────────

pub struct UniformBinding {
    pub buffer: wgpu::Buffer,
    pub bind_group: wgpu::BindGroup,
    pub layout: wgpu::BindGroupLayout,
}

impl UniformBinding {
    pub fn new(device: &wgpu::Device) -> Self {
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Uniform Buffer"),
            contents: bytemuck::cast_slice(&[UniformData::default()]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                count: None,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
            }],
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            }],
        });
        Self {
            buffer,
            bind_group,
            layout,
        }
    }

    pub fn write(&self, queue: &wgpu::Queue, data: UniformData) {
        queue.write_buffer(&self.buffer, 0, bytemuck::cast_slice(&[data]));
    }
}

// ── Scene ────────────────────────────────────────────────────────────────────

const VERTICES: [Vertex; 3] = [
    Vertex {
        position: [1.0, -1.0, 0.0, 1.0],
        color: [1.0, 0.0, 0.0, 1.0],
    },
    Vertex {
        position: [-1.0, -1.0, 0.0, 1.0],
        color: [0.0, 1.0, 0.0, 1.0],
    },
    Vertex {
        position: [0.0, 1.0, 0.0, 1.0],
        color: [0.0, 0.0, 1.0, 1.0],
    },
];
const INDICES: [u32; 3] = [0, 1, 2];

const SHADER: &str = "
struct Uniforms { mvp: mat4x4<f32> }
@group(0) @binding(0) var<uniform> u: Uniforms;

struct VIn  { @location(0) pos: vec4<f32>, @location(1) col: vec4<f32> }
struct VOut { @builtin(position) pos: vec4<f32>, @location(0) col: vec4<f32> }

@vertex   fn vs(v: VIn) -> VOut { return VOut(u.mvp * v.pos, v.col); }
@fragment fn fs(f: VOut) -> @location(0) vec4<f32> { return f.col; }
";

pub struct Scene {
    model: nalgebra_glm::Mat4,
    vbuf: wgpu::Buffer,
    ibuf: wgpu::Buffer,
    uniform: UniformBinding,
    pipeline: wgpu::RenderPipeline,
}

impl Scene {
    pub fn new(device: &wgpu::Device, surface_format: wgpu::TextureFormat) -> Self {
        let vbuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("VB"),
            contents: bytemuck::cast_slice(&VERTICES),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let ibuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("IB"),
            contents: bytemuck::cast_slice(&INDICES),
            usage: wgpu::BufferUsages::INDEX,
        });
        let uniform = UniformBinding::new(device);
        let pipeline = Self::build_pipeline(device, surface_format, &uniform);
        Self {
            model: nalgebra_glm::Mat4::identity(),
            vbuf,
            ibuf,
            uniform,
            pipeline,
        }
    }

    pub fn update(&mut self, queue: &wgpu::Queue, aspect: f32, dt: f32) {
        let proj = nalgebra_glm::perspective_lh_zo(aspect, 80_f32.to_radians(), 0.1, 1000.0);
        let view = nalgebra_glm::look_at_lh(
            &nalgebra_glm::vec3(0.0, 0.0, 3.0),
            &nalgebra_glm::vec3(0.0, 0.0, 0.0),
            &nalgebra_glm::Vec3::y(),
        );
        self.model = nalgebra_glm::rotate(
            &self.model,
            30_f32.to_radians() * dt,
            &nalgebra_glm::Vec3::y(),
        );
        self.uniform.write(
            queue,
            UniformData {
                mvp: proj * view * self.model,
            },
        );
    }

    pub fn render<'r>(&'r self, pass: &mut wgpu::RenderPass<'r>) {
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.uniform.bind_group, &[]);
        pass.set_vertex_buffer(0, self.vbuf.slice(..));
        pass.set_index_buffer(self.ibuf.slice(..), wgpu::IndexFormat::Uint32);
        pass.draw_indexed(0..INDICES.len() as u32, 0, 0..1);
    }

    fn build_pipeline(
        device: &wgpu::Device,
        fmt: wgpu::TextureFormat,
        uniform: &UniformBinding,
    ) -> wgpu::RenderPipeline {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&uniform.layout)],
            immediate_size: 0,
        });
        let attrs = Vertex::attrs();
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs"),
                buffers: &[Vertex::layout(&attrs)],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: fmt,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                strip_index_format: Some(wgpu::IndexFormat::Uint32),
                front_face: wgpu::FrontFace::Cw,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: Renderer::DEPTH_FORMAT,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview_mask: None,
            cache: None,
        })
    }
}

// ── GPU context ──────────────────────────────────────────────────────────────

pub struct Gpu {
    pub surface: wgpu::Surface<'static>,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub surface_config: wgpu::SurfaceConfiguration,
    pub surface_format: wgpu::TextureFormat,
}

impl Gpu {
    pub fn aspect_ratio(&self) -> f32 {
        self.surface_config.width as f32 / self.surface_config.height.max(1) as f32
    }

    pub fn resize(&mut self, w: u32, h: u32) {
        self.surface_config.width = w;
        self.surface_config.height = h;
        self.surface.configure(&self.device, &self.surface_config);
    }

    pub fn create_depth_texture(&self, w: u32, h: u32) -> wgpu::TextureView {
        self.device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("Depth Texture"),
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Depth32Float,
                size: wgpu::Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: 1,
                },
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
            .create_view(&wgpu::TextureViewDescriptor {
                format: Some(wgpu::TextureFormat::Depth32Float),
                dimension: Some(wgpu::TextureViewDimension::D2),
                aspect: wgpu::TextureAspect::All,
                ..Default::default()
            })
    }

    pub async fn new_async(
        window: impl Into<wgpu::SurfaceTarget<'static>>,
        w: u32,
        h: u32,
    ) -> Self {
        let instance =
            wgpu::Instance::new(InstanceDescriptor::new_without_display_handle_from_env());
        let surface = instance.create_surface(window).unwrap();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                compatible_surface: Some(&surface),
                ..Default::default()
            })
            .await
            .expect("No adapter");
        log::info!("WGPU Adapter Features: {:#?}", adapter.features());
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("Device"),
                required_limits: wgpu::Limits::default().using_resolution(adapter.limits()),
                ..Default::default()
            })
            .await
            .expect("No device");
        let caps = surface.get_capabilities(&adapter);
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|f| !f.is_srgb())
            .unwrap_or(caps.formats[0]);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: w,
            height: h,
            present_mode: caps.present_modes[0],
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);
        Self {
            surface,
            device,
            queue,
            surface_config: config,
            surface_format: format,
        }
    }
}

// ── Renderer ─────────────────────────────────────────────────────────────────

pub struct Renderer {
    gpu: Gpu,
    depth_view: wgpu::TextureView,
    egui_renderer: egui_wgpu::Renderer,
    scene: Scene,
}

impl Renderer {
    pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

    pub async fn new(window: impl Into<wgpu::SurfaceTarget<'static>>, w: u32, h: u32) -> Self {
        let gpu = Gpu::new_async(window, w, h).await;
        let depth_view = gpu.create_depth_texture(w, h);
        let egui_renderer = egui_wgpu::Renderer::new(
            &gpu.device,
            gpu.surface_config.format,
            egui_wgpu::RendererOptions {
                depth_stencil_format: Some(Self::DEPTH_FORMAT),
                msaa_samples: 1,
                ..Default::default()
            },
        );
        let scene = Scene::new(&gpu.device, gpu.surface_format);
        Self {
            gpu,
            depth_view,
            egui_renderer,
            scene,
        }
    }

    pub fn resize(&mut self, w: u32, h: u32) {
        self.gpu.resize(w, h);
        self.depth_view = self.gpu.create_depth_texture(w, h);
    }

    pub fn render_frame(
        &mut self,
        screen_desc: egui_wgpu::ScreenDescriptor,
        paint_jobs: Vec<egui::epaint::ClippedPrimitive>,
        textures_delta: egui::TexturesDelta,
        dt: Duration,
    ) {
        self.scene
            .update(&self.gpu.queue, self.gpu.aspect_ratio(), dt.as_secs_f32());

        for (id, delta) in &textures_delta.set {
            self.egui_renderer
                .update_texture(&self.gpu.device, &self.gpu.queue, *id, delta);
        }
        for id in &textures_delta.free {
            self.egui_renderer.free_texture(id);
        }

        let mut enc = self
            .gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Frame Encoder"),
            });
        self.egui_renderer.update_buffers(
            &self.gpu.device,
            &self.gpu.queue,
            &mut enc,
            &paint_jobs,
            &screen_desc,
        );

        let frame = match self.gpu.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(f)
            | wgpu::CurrentSurfaceTexture::Suboptimal(f) => f,
            wgpu::CurrentSurfaceTexture::Outdated => {
                self.gpu
                    .surface
                    .configure(&self.gpu.device, &self.gpu.surface_config);
                match self.gpu.surface.get_current_texture() {
                    wgpu::CurrentSurfaceTexture::Success(f)
                    | wgpu::CurrentSurfaceTexture::Suboptimal(f) => f,
                    other => panic!("Surface texture unavailable after reconfigure: {other:?}"),
                }
            }
            other => panic!("Surface texture unavailable: {other:?}"),
        };

        let surface_view = frame.texture.create_view(&wgpu::TextureViewDescriptor {
            format: Some(self.gpu.surface_format),
            ..Default::default()
        });

        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Main Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &surface_view,
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
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });
            self.scene.render(&mut pass);
            self.egui_renderer
                .render(&mut pass.forget_lifetime(), &paint_jobs, &screen_desc);
        }

        self.gpu.queue.submit(std::iter::once(enc.finish()));
        frame.present();
    }
}

// ── App (winit handler) ──────────────────────────────────────────────────────

#[derive(Default)]
pub struct App {
    window: Option<Arc<Window>>,
    renderer: Option<Renderer>,
    gui_state: Option<egui_winit::State>,
    last_render_time: Option<Instant>,
    last_size: (u32, u32),
    initialized: bool,
    #[cfg(target_arch = "wasm32")]
    renderer_receiver: Option<futures::channel::oneshot::Receiver<Renderer>>,
}

impl App {
    #[cfg(target_arch = "wasm32")]
    fn canvas() -> wgpu::web_sys::HtmlCanvasElement {
        wgpu::web_sys::window()
            .unwrap()
            .document()
            .unwrap()
            .get_element_by_id("canvas")
            .unwrap()
            .dyn_into::<wgpu::web_sys::HtmlCanvasElement>()
            .unwrap()
    }

    fn set_pixels_per_point(gui: &egui_winit::State, window: &Window) {
        #[cfg(not(target_arch = "wasm32"))]
        gui.egui_ctx()
            .set_pixels_per_point(window.scale_factor() as _);
        #[cfg(target_arch = "wasm32")]
        {
            let _ = window;
            gui.egui_ctx().set_pixels_per_point(1.0);
        }
    }
}

impl ApplicationHandler for App {
    fn suspended(&mut self, _: &winit::event_loop::ActiveEventLoop) {
        self.renderer = None;
        self.window = None;
    }

    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        let mut attrs = Window::default_attributes();

        #[cfg(not(target_arch = "wasm32"))]
        {
            attrs = attrs.with_title("Standalone Winit/Wgpu Example");
        }

        #[cfg(target_arch = "wasm32")]
        let (canvas_w, canvas_h);
        #[cfg(target_arch = "wasm32")]
        {
            use winit::platform::web::WindowAttributesExtWebSys;
            let canvas = Self::canvas();
            (canvas_w, canvas_h) = (canvas.width(), canvas.height());
            self.last_size = (canvas_w, canvas_h);
            attrs = attrs.with_canvas(Some(canvas));
        }

        let window = Arc::new(event_loop.create_window(attrs).unwrap());
        self.window = Some(window.clone());

        let gui_ctx = egui::Context::default();
        #[cfg(not(target_arch = "wasm32"))]
        {
            let s = window.inner_size();
            self.last_size = (s.width, s.height);
        }
        #[cfg(target_arch = "wasm32")]
        {
            gui_ctx.set_pixels_per_point(1.0);
        }

        let gui_state = egui_winit::State::new(
            gui_ctx,
            egui::Context::default().viewport_id(),
            &window,
            Some(window.scale_factor() as _),
            Some(Theme::Dark),
            None,
        );

        #[cfg(not(target_arch = "wasm32"))]
        {
            if !self.initialized {
                env_logger::init();
            }
            let (w, h) = self.last_size;
            self.renderer = Some(pollster::block_on(Renderer::new(window.clone(), w, h)));
        }
        #[cfg(target_arch = "wasm32")]
        {
            if !self.initialized {
                std::panic::set_hook(Box::new(console_error_panic_hook::hook));
                console_log::init().expect("logger init failed");
            }
            log::info!("Canvas dimensions: ({canvas_w} x {canvas_h})");
            let (tx, rx) = futures::channel::oneshot::channel();
            self.renderer_receiver = Some(rx);
            wasm_bindgen_futures::spawn_local(async move {
                if tx
                    .send(Renderer::new(window.clone(), canvas_w, canvas_h).await)
                    .is_err()
                {
                    log::error!("Failed to send renderer");
                }
            });
        }

        self.gui_state = Some(gui_state);
        self.last_render_time = Some(Instant::now());
        self.initialized = true;
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        _: winit::window::WindowId,
        event: WindowEvent,
    ) {
        #[cfg(target_arch = "wasm32")]
        if let Some(rx) = self.renderer_receiver.as_mut()
            && let Ok(Some(r)) = rx.try_recv()
        {
            self.renderer = Some(r);
            self.renderer_receiver = None;
            if let Some(w) = self.window.as_ref() {
                w.request_redraw();
            }
        }

        let (Some(gui), Some(renderer), Some(window), Some(last_time)) = (
            self.gui_state.as_mut(),
            self.renderer.as_mut(),
            self.window.as_ref(),
            self.last_render_time.as_mut(),
        ) else {
            return;
        };

        if gui.on_window_event(window, &event).consumed {
            return;
        }

        match event {
            WindowEvent::KeyboardInput {
                event:
                    winit::event::KeyEvent {
                        physical_key:
                            winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::Escape),
                        ..
                    },
                ..
            } => event_loop.exit(),
            WindowEvent::ScaleFactorChanged { .. } => Self::set_pixels_per_point(gui, window),
            WindowEvent::Resized(PhysicalSize {
                width: w,
                height: h,
            }) if w > 0 && h > 0 => {
                log::info!("Resize: ({w}, {h})");
                renderer.resize(w, h);
                self.last_size = (w, h);
                Self::set_pixels_per_point(gui, window);
            }
            WindowEvent::CloseRequested => {
                log::info!("Closing...");
                event_loop.exit();
            }
            WindowEvent::RedrawRequested => {
                #[cfg(target_arch = "wasm32")]
                {
                    let canvas = Self::canvas();
                    let (cw, ch) = (canvas.width(), canvas.height());
                    if (cw, ch) != self.last_size && cw > 0 && ch > 0 {
                        log::info!("Canvas resize: ({cw}, {ch})");
                        renderer.resize(cw, ch);
                        self.last_size = (cw, ch);
                    }
                }

                let now = Instant::now();
                let dt = now - *last_time;
                *last_time = now;

                #[cfg(not(target_arch = "wasm32"))]
                let gui_input = gui.take_egui_input(window);
                #[cfg(target_arch = "wasm32")]
                let mut gui_input = gui.take_egui_input(window);

                #[cfg(target_arch = "wasm32")]
                {
                    let canvas = Self::canvas();
                    let ppp = gui.egui_ctx().pixels_per_point();
                    let size =
                        egui::vec2(canvas.width() as f32 / ppp, canvas.height() as f32 / ppp);
                    gui_input.screen_rect = Some(egui::Rect::from_min_size(egui::Pos2::ZERO, size));
                }

                #[cfg(not(target_arch = "wasm32"))]
                let title = "Rust/Wgpu";
                #[cfg(target_arch = "wasm32")]
                let title = "Rust/Wgpu/WebGPU";

                let egui_winit::egui::FullOutput {
                    textures_delta,
                    shapes,
                    pixels_per_point,
                    platform_output,
                    ..
                } = gui.egui_ctx().run_ui(gui_input, |ui| {
                    egui::Panel::top("top").show_inside(ui, |ui| {
                        ui.horizontal(|ui| {
                            egui::MenuBar::new().ui(ui, |ui| {
                                ui.menu_button("File", |ui| {
                                    for label in ["Load", "Save"] {
                                        if ui.button(label).clicked() {
                                            ui.close();
                                        }
                                    }
                                    ui.separator();
                                    if ui.button("Import").clicked() {
                                        ui.close();
                                    }
                                });
                                ui.menu_button("Edit", |ui| {
                                    for label in ["Clear", "Reset"] {
                                        if ui.button(label).clicked() {
                                            ui.close();
                                        }
                                    }
                                });
                                ui.separator();
                                ui.label(
                                    egui::RichText::new(title).color(egui::Color32::LIGHT_GREEN),
                                );
                                ui.separator();
                            });
                            ui.with_layout(egui::Layout::right_to_left(egui::Align::RIGHT), |ui| {
                                ui.add_space(10.0);
                                ui.label(
                                    egui::RichText::new("v0.1.0").color(egui::Color32::ORANGE),
                                );
                                ui.separator();
                            });
                        });
                    });
                    egui::Panel::left("left").show_inside(ui, |ui| {
                        ui.heading("Scene Tree");
                    });
                    egui::Panel::right("right").show_inside(ui, |ui| {
                        ui.heading("Inspector");
                    });
                    egui::Panel::bottom("bottom").show_inside(ui, |ui| {
                        ui.heading("Console");
                    });
                });

                gui.handle_platform_output(window, platform_output);
                let paint_jobs = gui.egui_ctx().tessellate(shapes, pixels_per_point);

                let (w, h) = self.last_size;
                if w == 0 || h == 0 {
                    return;
                }

                renderer.render_frame(
                    egui_wgpu::ScreenDescriptor {
                        size_in_pixels: [w, h],
                        pixels_per_point,
                    },
                    paint_jobs,
                    textures_delta,
                    dt,
                );
            }
            _ => (),
        }

        window.request_redraw();
    }
}
