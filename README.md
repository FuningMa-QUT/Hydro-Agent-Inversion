**Hydro-Agent: Autonomous Inverse Modeling via Physics-Integrated LLM Agents**

This project provides Hydro-Agent, a physics-integrated multi-agent framework that leverages Large Language Models (LLMs) to automate groundwater inverse modeling and reactive transport calibration.

**🚀 Quick Navigation (For Reviewers)**

To facilitate the peer-review process, the following files provide the raw evidence for the qualitative reasoning and autonomous behavior discussed in the manuscript.

Full Execution Logs:Located at: outputs_and_logs

Contains the step-by-step decision-making process, prompt history, and autonomous code debugging logs.

**🛠️ Requirements**

Python 3.11 is highly recommended.

Dependencies are listed in requirements.txt.

**LLM Configuration**

This project uses LLMs (e.g., GPT-4o, DeepSeek) as reasoning engines. You must configure your own API key as a global environment variable:

export OPENAI_API_KEY='your-api-key-here'

**📂 Project Structure**

src/: Core logic of Hydro-Agent, including HydroCoder (autonomous code generator) and LogicRefiner.

cases/: Implementation scripts for Case I (Heterogeneity), Case II (Kinetics), and Case III (Aquia Aquifer).

outputs_and_logs/: Raw logs and generated reports demonstrating the agent's "semantic comprehension."

**📖 Citation**
If you use this project in your research, please cite:

Ma, F., Chen, J., Dai, Z., Cai, F., & Hu, Y. (2026). Autonomous inverse modeling of complex groundwater systems via a physics-integrated large language model multi-agent framework. Water Research.
