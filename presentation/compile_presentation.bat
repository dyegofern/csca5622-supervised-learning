@echo off
REM Cross-platform LaTeX Beamer Presentation Compiler for Windows
REM Usage: compile_presentation.bat <latex_file_name>
REM Example: compile_presentation.bat esg_presentation.tex

REM Check if parameter is provided
if "%~1"=="" (
    echo Error: No LaTeX file specified
    echo Usage: compile_presentation.bat ^<latex_file_name^>
    echo Example: compile_presentation.bat esg_presentation.tex
    exit /b 1
)

REM Get the filename and remove .tex extension if present
set "LATEX_FILE=%~1"
set "BASENAME=%~n1"

REM Check if file exists
if not exist "%BASENAME%.tex" (
    echo Error: File '%BASENAME%.tex' not found!
    exit /b 1
)

echo ==========================================
echo Compiling LaTeX Presentation: %BASENAME%.tex
echo ==========================================

REM First pass
echo Running first compilation pass...
pdflatex -interaction=nonstopmode "%BASENAME%.tex" >nul 2>&1

REM Check if PDF was created (not just exit code, as warnings can cause non-zero exit)
if not exist "%BASENAME%.pdf" (
    echo X First compilation pass failed - no PDF created!
    echo Check %BASENAME%.log for errors
    exit /b 1
)

REM Second pass (for TOC and references)
echo Running second compilation pass...
pdflatex -interaction=nonstopmode "%BASENAME%.tex" >nul 2>&1

REM Check if PDF still exists and was updated
if not exist "%BASENAME%.pdf" (
    echo X Second compilation pass failed!
    echo Check %BASENAME%.log for errors
    exit /b 1
)

echo.
echo ==========================================
echo Compilation complete!
echo Output: %BASENAME%.pdf
echo ==========================================

REM Optional: Clean up auxiliary files
REM Uncomment the following lines if you want to remove auxiliary files
REM echo Cleaning up auxiliary files...
REM del /Q "%BASENAME%.aux" "%BASENAME%.log" "%BASENAME%.nav" "%BASENAME%.out" "%BASENAME%.snm" "%BASENAME%.toc" "%BASENAME%.vrb" 2>nul

REM Open the PDF
echo.
echo Attempting to open PDF...
start "" "%BASENAME%.pdf"

exit /b 0
