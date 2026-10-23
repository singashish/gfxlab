@echo off
setlocal EnableExtensions

:: ---------------------------------------------------------------
:: Builds the Debug and Release configurations in one run, then
:: prints the commands to run the tests and the app with OpenGL
:: and Vulkan. Copy and run whichever you need.
::
:: Usage:  build.bat          (incremental build)
::         build.bat clean    (delete both build folders first)
::
:: Run it from a terminal that has the MSVC compiler (cl.exe),
:: for example an "x64 Native Tools Command Prompt for VS 2022".
:: ---------------------------------------------------------------

:: CUDA architecture (only used when the CUDA backend is found)
::   RTX 20 series: 75    RTX 30 series: 86
::   RTX 40 series: 89    RTX 50 series: 120
set "CUDA_ARCH=120"

:: Model used in the example commands (searched in assets\)
set "MODEL=cube.obj"

set "DEBUG_DIR=builds\build-debug"
set "RELEASE_DIR=builds\build-release"

:: Work from the folder that contains this script
cd /d "%~dp0"

:: Enable ANSI colors
for /F %%a in ('echo prompt $E ^| cmd') do set "ESC=%%a"

:: The MSVC compiler must be available
where cl >nul 2>nul
if errorlevel 1 (
    echo %ESC%[91m[X] cl.exe was not found.%ESC%[0m
    echo     Run this script from an x64 Native Tools Command Prompt for VS 2022,
    echo     or from a terminal that has the MSVC environment.
    exit /b 1
)

:: Optional clean build
if /i "%~1"=="clean" (
    echo %ESC%[96m[INFO] Removing the old build folders...%ESC%[0m
    if exist "%DEBUG_DIR%" rmdir /s /q "%DEBUG_DIR%"
    if exist "%RELEASE_DIR%" rmdir /s /q "%RELEASE_DIR%"
)

:: Build both configurations. Stop at the first failure
call :build Debug "%DEBUG_DIR%"
if errorlevel 1 exit /b 1

call :build Release "%RELEASE_DIR%"
if errorlevel 1 exit /b 1

:: Warn if the app was not produced
if not exist "%DEBUG_DIR%\src\app\gfxlab.exe" (
    echo %ESC%[93m[WARN] %DEBUG_DIR%\src\app\gfxlab.exe was not found%ESC%[0m
)
if not exist "%RELEASE_DIR%\src\app\gfxlab.exe" (
    echo %ESC%[93m[WARN] %RELEASE_DIR%\src\app\gfxlab.exe was not found%ESC%[0m
)

echo.
echo %ESC%[92m========================================%ESC%[0m
echo %ESC%[92m   Debug and Release builds finished%ESC%[0m
echo %ESC%[92m========================================%ESC%[0m
echo.
echo Copy and run any of these from the project folder:
echo.
echo %ESC%[96mDebug%ESC%[0m (Vulkan validation layer enabled)
echo   Tests:
echo     %ESC%[93mctest --test-dir %DEBUG_DIR% -C Debug --output-on-failure%ESC%[0m
echo   OpenGL:
echo     %ESC%[93m.\%DEBUG_DIR%\src\app\gfxlab.exe --backend=opengl %MODEL%%ESC%[0m
echo   Vulkan:
echo     %ESC%[93m.\%DEBUG_DIR%\src\app\gfxlab.exe --backend=vulkan %MODEL%%ESC%[0m
echo   Vulkan:
echo     %ESC%[93m.\%DEBUG_DIR%\src\app\gfxlab.exe --backend=cuda %MODEL%%ESC%[0m
echo.
echo %ESC%[96mRelease%ESC%[0m
echo   Tests:
echo     %ESC%[93mctest --test-dir %RELEASE_DIR% -C Release --output-on-failure%ESC%[0m
echo   OpenGL:
echo     %ESC%[93m.\%RELEASE_DIR%\src\app\gfxlab.exe --backend=opengl %MODEL%%ESC%[0m
echo   Vulkan:
echo     %ESC%[93m.\%RELEASE_DIR%\src\app\gfxlab.exe --backend=vulkan %MODEL%%ESC%[0m
echo   Cuda:
echo     %ESC%[93m.\%RELEASE_DIR%\src\app\gfxlab.exe --backend=cuda %MODEL%%ESC%[0m
echo.

endlocal
exit /b 0

:: ---------------------------------------------------------------
:: :build <Debug|Release> <build folder>
:: ---------------------------------------------------------------
:build
set "CONFIG=%~1"
set "DIR=%~2"

echo.
echo %ESC%[96m[INFO] Configuring %CONFIG%...%ESC%[0m
cmake -S . -B "%DIR%" -G Ninja -DCMAKE_BUILD_TYPE=%CONFIG% -DGFX_CUDA_ARCHITECTURES=%CUDA_ARCH%
if errorlevel 1 (
    echo %ESC%[91m[X] %CONFIG% configuration failed%ESC%[0m
    exit /b 1
)

echo.
echo %ESC%[96m[INFO] Building %CONFIG%...%ESC%[0m
cmake --build "%DIR%"
if errorlevel 1 (
    echo %ESC%[91m[X] %CONFIG% build failed%ESC%[0m
    exit /b 1
)

echo %ESC%[92m[OK] %CONFIG% build successful%ESC%[0m
exit /b 0