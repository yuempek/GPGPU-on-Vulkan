@echo off
"%~dp0..\zig\zig.exe" cc -target x86_64-linux-gnu %*
