@echo off
"%~dp0..\zig\zig.exe" c++ -target aarch64-linux-gnu %*
