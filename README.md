cbf_qp/
├── scenarios/
│   ├── ZAM_Zip-1_64_T-1.xml       # CommonRoad XML scenario
│   ├── USA_US101-9_1_T-1.xml  
│   └── ZAM_Zip-1_32_I-1-1.xml
├── src/
│   ├── __init__.py
│   ├── scenario_loader.py          # Scenario parsing & ego/obstacle extraction
│   ├── cbf_solver.py               # CBF formulation & QP optimization class
│   ├── vehicle_dynamics.py         # Kinematic updates & geometry utilities
│   └── visualizer.py               # Matplotlib rendering & GIF generation
├── main.py                         # Clean entry point orchestrating the loop
├── requirements.txt
└── README.md
