@echo off
REM ============================================================
REM  Full Experiment Pipeline
REM  Run from project root: scripts\run_all_experiments.bat
REM ============================================================

echo ============================================================
echo   ADVERSARIAL ML SECURITY FRAMEWORK - FULL EXPERIMENT RUN
echo ============================================================

echo.
echo [1/7] Generating training data...
python src\simulation\traffic_generator.py
python src\simulation\attack_generator.py

echo.
echo [2/7] Training baseline models...
python src\training\core_model.py

echo.
echo [3/7] Multi-seed evaluation...
python scripts\train_multiseed.py

echo.
echo [4/7] Adversarial training comparison...
python scripts\train_adversarial.py

echo.
echo [5/7] ROC curve and threshold analysis...
python scripts\generate_roc.py

echo.
echo [6/7] Epsilon sweep experiment...
python scripts\run_epsilon_sweep.py

echo.
echo [7/7] Zero-Trust ablation study...
python scripts\run_ablation.py

echo.
echo ============================================================
echo   ALL EXPERIMENTS COMPLETE
echo   Results: results\
echo   Figures: figures\
echo ============================================================
pause
