# System architecture

![architecture](architecture.png)

```mermaid
flowchart LR
  subgraph Input
    RGB[HR RGB guide]
    LR[LR / sparse / noisy depth]
  end
  subgraph Analyse["adaptive.py: input analysis"]
    A1[scale, noise, hole ratio,<br/>texture-copy risk]
    A2[decisions: denoise? guided?<br/>dual_path? refine?]
  end
  subgraph Classical["classical.py"]
    C1[bilateral denoise] --> C2[large-hole fill] --> C3[Joint Bilateral Upsampling] --> C4[guided filter]
    B0[hole fill + bicubic]
  end
  subgraph Neural["model.py: DepthForge-Net 0.5M, tiled 128² inference"]
    N1[CNN on bicubic prior]
    N2[CNN on classical prior]
  end
  subgraph Hybrid["pipeline.py: hybrid"]
    F1{many holes?}
    F2[average both CNN outputs]
    F3[edge-aware guided refinement<br/>if 8x or holes]
  end
  RGB --> Analyse
  LR --> Analyse
  A1 --> A2
  A2 -.params.-> Classical
  A2 -.decisions.-> Hybrid
  LR --> C1
  LR --> B0
  B0 --> N1
  C4 --> N2
  RGB --> C3 & C4 & N1 & N2 & F3
  N1 --> F1
  F1 -- yes --> F2
  N2 --> F2
  F1 -- no --> F3
  F2 --> F3 --> OUT[HR depth + metrics + viz]
  subgraph Serving
    APP[Gradio switchboard app.py<br/>Docker / HF Spaces]
    CLI[JSON CLI depthforge.cli]
  end
  OUT --> APP
  OUT --> CLI
  subgraph Switch["Switch room: depthforge-lab (Discord/Telegram)"]
    L[forge-lead] --> CL[forge-classical]
    L --> NN[forge-neural]
    L --> J[forge-judge]
    H((human approves))
  end
  CL & NN & J -- run --> CLI
  H -.-> L
```

## Code ↔ diagram
| Block | File / function |
|---|---|
| Input analysis | `adaptive.analyze`, `adaptive.choose_params` |
| Denoise | `classical.denoise_depth` |
| Hole fill | `classical.fill_large_holes`, `classical.fill_holes` |
| JBU | `classical.joint_bilateral_upsample` |
| Guided filter | `classical.guided_filter` |
| CNN (tiled inference) | `model.DepthForgeNet`, `pipeline.run_cnn` |
| Hybrid selection / dual-path fusion / refinement | `pipeline.run` (hybrid branch), `adaptive.choose_params` (`dual_path`, `refine`) |
| Discontinuities + snapping (opt-in) | `classical.depth_discontinuities`, `classical.snap_discontinuities` |
| Metrics | `metrics.all_metrics` (RMSE, MAE, PSNR, SSIM, Edge-RMSE, Boundary-F1) + latency/memory in `pipeline.run` |
| Serving | `app.py`, `depthforge/cli.py`, `Dockerfile` |
| Agents | `switch/depthforge-lab.template.yaml`, `switch/AGENTS.md` |
