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

const TRI: [([f32; 3], [f32; 3]); 3] = [
    ([1., -1., 0.], [1., 0., 0.]),
    ([-1., -1., 0.], [0., 1., 0.]),
    ([0., 1., 0.], [0., 0., 1.]),
];
const SHADER: &str = "
struct O{@builtin(position)p:vec4<f32>,@location(0)c:vec4<f32>}
@vertex fn vs(@location(0)p:vec4<f32>,@location(1)c:vec4<f32>)->O{return O(p,c);}
@fragment fn fs(o:O)->@location(0)vec4<f32>{return o.c;}
";

#[cfg(target_arch = "wasm32")]
fn cv() -> wgpu::web_sys::HtmlCanvasElement {
    wgpu::web_sys::window()
        .unwrap()
        .document()
        .unwrap()
        .get_element_by_id("canvas")
        .unwrap()
        .dyn_into()
        .unwrap()
}

struct G {
    w: Arc<Window>,
    s: wgpu::Surface<'static>,
    d: wgpu::Device,
    q: wgpu::Queue,
    cfg: wgpu::SurfaceConfiguration,
    p: wgpu::RenderPipeline,
    vb: wgpu::Buffer,
    er: egui_wgpu::Renderer,
    es: egui_winit::State,
    m: nalgebra_glm::Mat4,
    t: Instant,
    sz: (u32, u32),
}

#[derive(Default)]
pub struct App {
    g: Option<G>,
    #[cfg(target_arch = "wasm32")]
    rx: Option<futures::channel::oneshot::Receiver<G>>,
}

async fn init(w: Arc<Window>, aw: u32, ah: u32) -> G {
    let i = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
    let s = i.create_surface(w.clone()).unwrap();
    let a = i
        .request_adapter(&wgpu::RequestAdapterOptions {
            compatible_surface: Some(&s),
            ..Default::default()
        })
        .await
        .unwrap();
    let (d, q) = a
        .request_device(&wgpu::DeviceDescriptor {
            required_limits: wgpu::Limits::default().using_resolution(a.limits()),
            ..Default::default()
        })
        .await
        .unwrap();
    let cfg = s.get_default_config(&a, aw, ah).unwrap();
    s.configure(&d, &cfg);
    let sh = d.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(SHADER.into()),
    });
    let at = wgpu::vertex_attr_array![0=>Float32x4,1=>Float32x4];
    let p = d.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: None,
        layout: None,
        vertex: wgpu::VertexState {
            module: &sh,
            entry_point: Some("vs"),
            compilation_options: Default::default(),
            buffers: &[wgpu::VertexBufferLayout {
                array_stride: 32,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &at,
            }],
        },
        fragment: Some(wgpu::FragmentState {
            module: &sh,
            entry_point: Some("fs"),
            compilation_options: Default::default(),
            targets: &[Some(cfg.format.into())],
        }),
        primitive: Default::default(),
        depth_stencil: None,
        multisample: Default::default(),
        multiview_mask: None,
        cache: None,
    });
    let vb = d.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 96,
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let er = egui_wgpu::Renderer::new(
        &d,
        cfg.format,
        egui_wgpu::RendererOptions {
            msaa_samples: 1,
            ..Default::default()
        },
    );
    let ctx = egui::Context::default();
    let es = egui_winit::State::new(ctx, egui::ViewportId::ROOT, &w, None, None, None);
    G {
        w,
        s,
        d,
        q,
        cfg,
        p,
        vb,
        er,
        es,
        m: nalgebra_glm::Mat4::identity(),
        t: Instant::now(),
        sz: (aw, ah),
    }
}

fn resize(g: &mut G, w: u32, h: u32) {
    (g.sz, g.cfg.width, g.cfg.height) = ((w, h), w, h);
    g.s.configure(&g.d, &g.cfg);
}

fn render(g: &mut G) {
    let now = Instant::now();
    let dt = (now - g.t).as_secs_f32();
    g.t = now;
    #[cfg(target_arch = "wasm32")]
    {
        let c = cv();
        if c.width() > 0 && c.height() > 0 && (c.width(), c.height()) != g.sz {
            resize(g, c.width(), c.height());
        }
    }
    let (w, h) = g.sz;
    #[cfg(not(target_arch = "wasm32"))]
    let inp = g.es.take_egui_input(&g.w);
    #[cfg(target_arch = "wasm32")]
    let mut inp = g.es.take_egui_input(&g.w);
    #[cfg(target_arch = "wasm32")]
    {
        let ppp = g.es.egui_ctx().pixels_per_point();
        inp.screen_rect = Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(w as f32 / ppp, h as f32 / ppp),
        ));
    }
    let o = g.es.egui_ctx().run_ui(inp, |ui| {
        egui::Window::new("wgpu+egui").show(ui.ctx(), |ui| {
            ui.label("Spinning triangle");
            ui.label(format!("{:.0}fps", 1. / dt.max(1e-6)));
        });
    });
    g.es.handle_platform_output(&g.w, o.platform_output);
    let jobs = g.es.egui_ctx().tessellate(o.shapes, o.pixels_per_point);
    let dsc = egui_wgpu::ScreenDescriptor {
        size_in_pixels: [w, h],
        pixels_per_point: o.pixels_per_point,
    };
    for (id, d) in &o.textures_delta.set {
        g.er.update_texture(&g.d, &g.q, *id, d);
    }
    for id in &o.textures_delta.free {
        g.er.free_texture(id);
    }
    g.m = nalgebra_glm::rotate(&g.m, 30f32.to_radians() * dt, &nalgebra_glm::Vec3::y());
    let mvp =
        nalgebra_glm::perspective_lh_zo(w as f32 / h.max(1) as f32, 80f32.to_radians(), 0.1, 1e3)
            * nalgebra_glm::look_at_lh(
                &nalgebra_glm::vec3(0., 0., 3.),
                &nalgebra_glm::vec3(0., 0., 0.),
                &nalgebra_glm::Vec3::y(),
            )
            * g.m;
    let mut vd = [0f32; 24];
    for (i, (p, c)) in TRI.iter().enumerate() {
        let v = mvp * nalgebra_glm::vec4(p[0], p[1], p[2], 1.);
        vd[i * 8..i * 8 + 8].copy_from_slice(&[v.x, v.y, v.z, v.w, c[0], c[1], c[2], 1.]);
    }
    g.q.write_buffer(&g.vb, 0, bytemuck::cast_slice(&vd));
    let mut enc = g.d.create_command_encoder(&Default::default());
    g.er.update_buffers(&g.d, &g.q, &mut enc, &jobs, &dsc);
    let frame = loop {
        match g.s.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(f)
            | wgpu::CurrentSurfaceTexture::Suboptimal(f) => break f,
            wgpu::CurrentSurfaceTexture::Outdated => g.s.configure(&g.d, &g.cfg),
            o => panic!("{o:?}"),
        }
    };
    let tv = frame.texture.create_view(&Default::default());
    {
        let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &tv,
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
        pass.set_pipeline(&g.p);
        pass.set_vertex_buffer(0, g.vb.slice(..));
        pass.draw(0..3, 0..1);
        g.er.render(&mut pass.forget_lifetime(), &jobs, &dsc);
    }
    g.q.submit([enc.finish()]);
    frame.present();
}

impl ApplicationHandler for App {
    fn resumed(&mut self, el: &ActiveEventLoop) {
        if self.g.is_some() {
            return;
        }
        #[cfg(not(target_arch = "wasm32"))]
        let attrs = Window::default_attributes();
        #[cfg(target_arch = "wasm32")]
        let (attrs, cw, ch) = {
            use winit::platform::web::WindowAttributesExtWebSys;
            let c = cv();
            let (w, h) = (c.width(), c.height());
            (Window::default_attributes().with_canvas(Some(c)), w, h)
        };
        let win = Arc::new(el.create_window(attrs).unwrap());
        #[cfg(not(target_arch = "wasm32"))]
        {
            let _ = env_logger::try_init();
            let s = win.inner_size();
            self.g = Some(pollster::block_on(init(win, s.width, s.height)));
        }
        #[cfg(target_arch = "wasm32")]
        {
            console_error_panic_hook::set_once();
            let _ = console_log::init();
            let (tx, rx) = futures::channel::oneshot::channel();
            self.rx = Some(rx);
            wasm_bindgen_futures::spawn_local(async move {
                let _ = tx.send(init(win, cw, ch).await);
            });
        }
    }
    fn window_event(&mut self, el: &ActiveEventLoop, _: WindowId, ev: WindowEvent) {
        #[cfg(target_arch = "wasm32")]
        if let Some(rx) = self.rx.as_mut()
            && let Ok(Some(g)) = rx.try_recv()
        {
            g.w.request_redraw();
            self.g = Some(g);
            self.rx = None;
        }
        let Some(g) = self.g.as_mut() else { return };
        if g.es.on_window_event(&g.w, &ev).consumed {
            g.w.request_redraw();
            return;
        }
        match ev {
            WindowEvent::CloseRequested
            | WindowEvent::KeyboardInput {
                event:
                    winit::event::KeyEvent {
                        physical_key: PhysicalKey::Code(KeyCode::Escape),
                        ..
                    },
                ..
            } => el.exit(),
            WindowEvent::Resized(s) if s.width > 0 && s.height > 0 => resize(g, s.width, s.height),
            WindowEvent::RedrawRequested => {
                render(g);
                g.w.request_redraw();
            }
            _ => {}
        }
    }
}
