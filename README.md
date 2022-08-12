# ModewiseTRIP

This contains the code and results needed to reproduce the figures in the (revised) paper "Modewise Operators, the Tensor Restricted Isometry Property, and Low-Rank Tensor Recovery"

See: https://export.arxiv.org/abs/2109.10454

The following command will run two-step modewise measurement and recovery for a four mode $40\times40\times40\times40$ tensor at ranks 3 and 4 with an intermediary skethcing dimension of $m_1 = 300$ and a final sketching dimension of $m_0 = 1024$, performing ten trials with SORS measurement matrices (subsampled Fourier transform).

python run_mw_trial.py "40" "4" "3,5" "300" "1024" "10" "TWOSTEP" "Fourier"

Figures can be preproduced using the notebook
