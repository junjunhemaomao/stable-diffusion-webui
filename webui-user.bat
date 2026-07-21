@echo off

set PYTHON=
set GIT=
set VENV_DIR=
COMMANDLINE_ARGS=--xformers --bf16
set WEBUI_LAUNCH_LIVE_OUTPUT=1

call webui.bat
