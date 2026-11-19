@echo off
"%~dp0..\zig\zig.exe" c++ -target x86_64-linux-gnu %*
