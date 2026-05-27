use std::sync::Arc;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;
use web_time::Instant;
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::ActiveEventLoop,
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowId},
};

const TRIANGLE_VERTICES: [([f32; 3], [f32; 3]); 3] = [
    ([1., -1., 0.], [1., 0., 0.]),
    ([-1., -1., 0.], [0., 1., 0.]),
    ([0., 1., 0.], [0., 0., 1.]),
];

const SHADER: &str = "
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
}
@vertex fn vs(@location(0) position: vec4<f32>, @location(1) color: vec4<f32>) -> VertexOutput {
    return VertexOutput(position, color);
}
@fragment fn fs(in: VertexOutput) -> @location(0) vec4<f32> {
    return in.color;
}
";

#[cfg(target_arch = "wasm32")]
fn get_canvas() -> wgpu::web_sys::HtmlCanvasElement {
    wgpu::web_sys::window()
        .unwrap()
        .document()
        .unwrap()
        .get_element_by_id("canvas")
        .unwrap()
        .dyn_into()
        .unwrap()
}

struct Graphics {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface_config: wgpu::SurfaceConfiguration,
    pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    egui_renderer: egui_wgpu::Renderer,
    egui_state: egui_winit::State,
    rotation: nalgebra_glm::Mat4,
    last_frame: Instant,
    size: (u32, u32),
}

#[derive(Default)]
pub struct App {
    graphics: Option<Graphics>,
    #[cfg(target_arch = "wasm32")]
    pending: Option<futures::channel::oneshot::Receiver<Graphics>>,
}

async fn init_graphics(window: Arc<Window>, width: u32, height: u32) -> Graphics {
    let instance =
        wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
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
    let surface_config = surface.get_default_config(&adapter, width, height).unwrap();
    surface.configure(&device, &surface_config);

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(SHADER.into()),
    });
    let vertex_attrs = wgpu::vertex_attr_array![0 => Float32x4, 1 => Float32x4];
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
                attributes: &vertex_attrs,
            }],
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs"),
            compilation_options: Default::default(),
            targets: &[Some(surface_config.format.into())],
        }),
        primitive: Default::default(),
        depth_stencil: None,
        multisample: Default::default(),
        multiview_mask: None,
        cache: None,
    });

    let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 96,
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let egui_renderer = egui_wgpu::Renderer::new(
        &device,
        surface_config.format,
        egui_wgpu::RendererOptions {
            msaa_samples: 1,
            ..Default::default()
        },
    );
    let egui_context = egui::Context::default();
    let egui_state = egui_winit::State::new(
        egui_context,
        egui::ViewportId::ROOT,
        &window,
        None,
        None,
        None,
    );

    Graphics {
        window,
        surface,
        device,
        queue,
        surface_config,
        pipeline,
        vertex_buffer,
        egui_renderer,
        egui_state,
        rotation: nalgebra_glm::Mat4::identity(),
        last_frame: Instant::now(),
        size: (width, height),
    }
}

fn resize(graphics: &mut Graphics, width: u32, height: u32) {
    graphics.size = (width, height);
    graphics.surface_config.width = width;
    graphics.surface_config.height = height;
    graphics
        .surface
        .configure(&graphics.device, &graphics.surface_config);
}

fn render(graphics: &mut Graphics) {
    let now = Instant::now();
    let dt = (now - graphics.last_frame).as_secs_f32();
    graphics.last_frame = now;

    #[cfg(target_arch = "wasm32")]
    {
        let canvas = get_canvas();
        if canvas.width() > 0
            && canvas.height() > 0
            && (canvas.width(), canvas.height()) != graphics.size
        {
            resize(graphics, canvas.width(), canvas.height());
        }
    }

    let (width, height) = graphics.size;

    #[cfg(not(target_arch = "wasm32"))]
    let egui_input = graphics.egui_state.take_egui_input(&graphics.window);
    #[cfg(target_arch = "wasm32")]
    let mut egui_input = graphics.egui_state.take_egui_input(&graphics.window);
    #[cfg(target_arch = "wasm32")]
    {
        let ppp = graphics.egui_state.egui_ctx().pixels_per_point();
        egui_input.screen_rect = Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(width as f32 / ppp, height as f32 / ppp),
        ));
    }

    let egui_output = graphics.egui_state.egui_ctx().run_ui(egui_input, |ui| {
        egui::Window::new("wgpu+egui").show(ui.ctx(), |ui| {
            ui.label("Spinning triangle");
            ui.label(format!("{:.0}fps", 1. / dt.max(1e-6)));
        });
    });
    graphics
        .egui_state
        .handle_platform_output(&graphics.window, egui_output.platform_output);

    let paint_jobs = graphics
        .egui_state
        .egui_ctx()
        .tessellate(egui_output.shapes, egui_output.pixels_per_point);
    let screen_descriptor = egui_wgpu::ScreenDescriptor {
        size_in_pixels: [width, height],
        pixels_per_point: egui_output.pixels_per_point,
    };

    for (id, delta) in &egui_output.textures_delta.set {
        graphics
            .egui_renderer
            .update_texture(&graphics.device, &graphics.queue, *id, delta);
    }
    for id in &egui_output.textures_delta.free {
        graphics.egui_renderer.free_texture(id);
    }

    graphics.rotation = nalgebra_glm::rotate(
        &graphics.rotation,
        30f32.to_radians() * dt,
        &nalgebra_glm::Vec3::y(),
    );
    let mvp = nalgebra_glm::perspective_lh_zo(
        width as f32 / height.max(1) as f32,
        80f32.to_radians(),
        0.1,
        1e3,
    ) * nalgebra_glm::look_at_lh(
        &nalgebra_glm::vec3(0., 0., 3.),
        &nalgebra_glm::vec3(0., 0., 0.),
        &nalgebra_glm::Vec3::y(),
    ) * graphics.rotation;

    let mut vertex_data = [0f32; 24];
    for (i, (pos, color)) in TRIANGLE_VERTICES.iter().enumerate() {
        let clip = mvp * nalgebra_glm::vec4(pos[0], pos[1], pos[2], 1.);
        vertex_data[i * 8..i * 8 + 8].copy_from_slice(&[
            clip.x, clip.y, clip.z, clip.w, color[0], color[1], color[2], 1.,
        ]);
    }
    graphics.queue.write_buffer(
        &graphics.vertex_buffer,
        0,
        bytemuck::cast_slice(&vertex_data),
    );

    let mut encoder = graphics.device.create_command_encoder(&Default::default());
    graphics.egui_renderer.update_buffers(
        &graphics.device,
        &graphics.queue,
        &mut encoder,
        &paint_jobs,
        &screen_descriptor,
    );

    let frame = loop {
        match graphics.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(f)
            | wgpu::CurrentSurfaceTexture::Suboptimal(f) => break f,
            wgpu::CurrentSurfaceTexture::Outdated => graphics
                .surface
                .configure(&graphics.device, &graphics.surface_config),
            other => panic!("{other:?}"),
        }
    };
    let frame_view = frame.texture.create_view(&Default::default());
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &frame_view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.19,
                        g: 0.24,
                        b: 0.42,
                        a: 1.,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            ..Default::default()
        });
        pass.set_pipeline(&graphics.pipeline);
        pass.set_vertex_buffer(0, graphics.vertex_buffer.slice(..));
        pass.draw(0..3, 0..1);
        graphics
            .egui_renderer
            .render(&mut pass.forget_lifetime(), &paint_jobs, &screen_descriptor);
    }
    graphics.queue.submit([encoder.finish()]);
    frame.present();
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.graphics.is_some() {
            return;
        }

        #[cfg(not(target_arch = "wasm32"))]
        let window_attrs = Window::default_attributes();
        #[cfg(target_arch = "wasm32")]
        let (window_attrs, canvas_width, canvas_height) = {
            use winit::platform::web::WindowAttributesExtWebSys;
            let canvas = get_canvas();
            let (w, h) = (canvas.width(), canvas.height());
            (Window::default_attributes().with_canvas(Some(canvas)), w, h)
        };

        let window = Arc::new(event_loop.create_window(window_attrs).unwrap());

        #[cfg(not(target_arch = "wasm32"))]
        {
            let _ = env_logger::try_init();
            let size = window.inner_size();
            self.graphics = Some(pollster::block_on(init_graphics(
                window,
                size.width,
                size.height,
            )));
        }
        #[cfg(target_arch = "wasm32")]
        {
            console_error_panic_hook::set_once();
            let _ = console_log::init();
            let (sender, receiver) = futures::channel::oneshot::channel();
            self.pending = Some(receiver);
            wasm_bindgen_futures::spawn_local(async move {
                let _ = sender.send(init_graphics(window, canvas_width, canvas_height).await);
            });
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        #[cfg(target_arch = "wasm32")]
        if let Some(receiver) = self.pending.as_mut()
            && let Ok(Some(graphics)) = receiver.try_recv()
        {
            graphics.window.request_redraw();
            self.graphics = Some(graphics);
            self.pending = None;
        }

        let Some(graphics) = self.graphics.as_mut() else {
            return;
        };

        if graphics
            .egui_state
            .on_window_event(&graphics.window, &event)
            .consumed
        {
            graphics.window.request_redraw();
            return;
        }

        match event {
            WindowEvent::CloseRequested
            | WindowEvent::KeyboardInput {
                event:
                    winit::event::KeyEvent {
                        physical_key: PhysicalKey::Code(KeyCode::Escape),
                        ..
                    },
                ..
            } => event_loop.exit(),
            WindowEvent::Resized(size) if size.width > 0 && size.height > 0 => {
                resize(graphics, size.width, size.height);
            }
            WindowEvent::RedrawRequested => {
                render(graphics);
                graphics.window.request_redraw();
            }
            _ => {}
        }
    }
}
