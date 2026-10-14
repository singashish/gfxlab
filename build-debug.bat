@echo off
setlocal

:: Enable ANSI colors
for /F %%a in ('echo prompt $E ^| cmd') do set "ESC=%%a"

:: Debug build directory
set "BUILD_DIR=builds\build-debug"

:: Delete the build-debug directory
rmdir /s /q builds\build-debug

:: Configure project
echo %ESC%[96m[INFO] Configuring Debug build...%ESC%[0m

:: RTX 20 series
:: cmake -S . -B %BUILD_DIR% -G Ninja -DCMAKE_BUILD_TYPE=Debug -DGFX_CUDA_ARCHITECTURES=75

:: RTX 30 series
:: cmake -S . -B %BUILD_DIR% -G Ninja -DCMAKE_BUILD_TYPE=Debug -DGFX_CUDA_ARCHITECTURES=86

:: RTX 40 series
:: cmake -S . -B %BUILD_DIR% -G Ninja -DCMAKE_BUILD_TYPE=Debug -DGFX_CUDA_ARCHITECTURES=89

:: RTX 50 series, including RTX 5080
cmake -S . -B %BUILD_DIR% -G Ninja -DCMAKE_BUILD_TYPE=Debug -DGFX_CUDA_ARCHITECTURES=120

:: Build for several GPU generations
:: cmake -S . -B %BUILD_DIR% -G Ninja -DCMAKE_BUILD_TYPE=Debug "-DGFX_CUDA_ARCHITECTURES=75;86;89;120"

if errorlevel 1 (
    echo %ESC%[91m[X] CMake configuration failed%ESC%[0m
    pause
    exit /b 1
)

:: Build project
echo.
echo %ESC%[96m[INFO] Building Debug configuration...%ESC%[0m
cmake --build %BUILD_DIR% --config Debug

if errorlevel 1 (
    echo %ESC%[91m[X] Debug build failed%ESC%[0m
    pause
    exit /b 1
)

echo.
echo %ESC%[92m[OK] Debug build successful!%ESC%[0m

:: Executable selection menu
:menu
echo.
echo %ESC%[96m========================================%ESC%[0m
echo %ESC%[96m       GFXLAB DEBUG EXECUTION MENU%ESC%[0m
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

:run_gfxlab
echo.
echo %ESC%[96m[INFO] Launching Debug gfxlab.exe...%ESC%[0m
echo.

if exist "%BUILD_DIR%\src\app\gfxlab.exe" (
    "%BUILD_DIR%\src\app\gfxlab.exe"
) else if exist "%BUILD_DIR%\src\app\Debug\gfxlab.exe" (
    "%BUILD_DIR%\src\app\Debug\gfxlab.exe"
) else (
    echo %ESC%[91m[X] gfxlab.exe not found!%ESC%[0m
)
goto after_execution

:run_test_math
echo.
echo %ESC%[96m[INFO] Launching Debug test_math.exe...%ESC%[0m
echo.

if exist "%BUILD_DIR%\tests\test_math.exe" (
    "%BUILD_DIR%\tests\test_math.exe"
    if errorlevel 1 (
        echo.
        echo %ESC%[91m[FAIL] Math tests failed%ESC%[0m
    ) else (
        echo.
        echo %ESC%[92m[OK] Math tests passed%ESC%[0m
    )
) else if exist "%BUILD_DIR%\tests\Debug\test_math.exe" (
    "%BUILD_DIR%\tests\Debug\test_math.exe"
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

:run_ctest
echo.
echo %ESC%[96m[INFO] Running Debug tests with CTest...%ESC%[0m
echo.

ctest --test-dir %BUILD_DIR% -C Debug --output-on-failure

if errorlevel 1 (
    echo.
    echo %ESC%[91m[FAIL] One or more tests failed%ESC%[0m
) else (
    echo.
    echo %ESC%[92m[OK] All tests passed%ESC%[0m
)
goto after_execution

:after_execution
echo.
pause
goto menu

:exit_script
echo.
echo %ESC%[93m[INFO] Exiting...%ESC%[0m
echo.
endlocal
exit /b 0
