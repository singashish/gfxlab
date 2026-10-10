@echo off
setlocal

:: Delete the build directory
rmdir /s /q build

:: Enable ANSI colors
for /F %%a in ('echo prompt $E ^| cmd') do set "ESC=%%a"

:: Configure project
echo %ESC%[96m[INFO] Configuring project...%ESC%[0m
:: RTX 20 series
:: cmake -S . -B build -G Ninja -DGFX_CUDA_ARCHITECTURES=75

:: RTX 30 series
:: cmake -S . -B build -G Ninja -DGFX_CUDA_ARCHITECTURES=86

:: RTX 40 series
:: cmake -S . -B build -G Ninja -DGFX_CUDA_ARCHITECTURES=89

:: RTX 50 series, including your RTX 5080
cmake -S . -B build -G Ninja -DGFX_CUDA_ARCHITECTURES=120

:: Build for several GPU generations
:: cmake -S . -B build -G Ninja "-DGFX_CUDA_ARCHITECTURES=75;86;89;120"

if errorlevel 1 (
    echo %ESC%[91m[X] CMake configuration failed%ESC%[0m
    pause
    exit /b 1
)

:: Build project
echo.
echo %ESC%[96m[INFO] Building project...%ESC%[0m
cmake --build build

if errorlevel 1 (
    echo %ESC%[91m[X] Build failed%ESC%[0m
    pause
    exit /b 1
)

echo.
echo %ESC%[92m[OK] Build successful!%ESC%[0m

:: Executable selection menu
:menu
echo.
echo %ESC%[96m========================================%ESC%[0m
echo %ESC%[96m          GFXLAB EXECUTION MENU%ESC%[0m
echo %ESC%[96m========================================%ESC%[0m
echo.
echo   %ESC%[93m[0]%ESC%[0m Exit
echo   %ESC%[93m[1]%ESC%[0m Run gfxlab.exe
echo   %ESC%[93m[2]%ESC%[0m Run test_math.exe
echo   %ESC%[93m[3]%ESC%[0m Run all tests with CTest
echo.

set "choice="
set /p "choice=Enter your choice: "

if "%choice%"=="0" goto exit_script
if "%choice%"=="1" goto run_gfxlab
if "%choice%"=="2" goto run_test_math
if "%choice%"=="3" goto run_ctest

echo.
echo %ESC%[91m[X] Invalid choice. Please try again%ESC%[0m
goto menu

:: Run gfxlab
:run_gfxlab
echo.
echo %ESC%[96m[INFO] Launching gfxlab.exe...%ESC%[0m
echo.

if exist ".\build\src\app\gfxlab.exe" (
    ".\build\src\app\gfxlab.exe"
) else (
    echo %ESC%[91m[X] gfxlab.exe not found!%ESC%[0m
)
goto after_execution

:: Run math tests directly
:run_test_math
echo.
echo %ESC%[96m[INFO] Launching test_math.exe...%ESC%[0m
echo.

if exist ".\build\tests\test_math.exe" (
    ".\build\tests\test_math.exe"
    if errorlevel 1 (
        echo.
        echo %ESC%[91m[FAIL] Math tests failed%ESC%[0m
    ) else (
        echo.
        echo %ESC%[92m[OK] Math tests passed%ESC%[0m
    )
) else (
    echo %ESC%[91m[X] test_math.exe not found!%ESC%[0m
)
goto after_execution

:: Run all registered tests with CTest
:run_ctest
echo.
echo %ESC%[96m[INFO] Running all tests with CTest...%ESC%[0m
echo.

ctest --test-dir build --output-on-failure

if errorlevel 1 (
    echo.
    echo %ESC%[91m[FAIL] One or more tests failed%ESC%[0m
) else (
    echo.
    echo %ESC%[92m[OK] All tests passed%ESC%[0m
)
goto after_execution

:: Return to menu after execution
:after_execution
echo.
pause
goto menu

:: Exit
:exit_script
echo.
echo %ESC%[93m[INFO] Exiting...%ESC%[0m
echo.
endlocal
exit /b 0
