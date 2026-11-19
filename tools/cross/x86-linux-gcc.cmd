@echo off
"%~dp0..\zig\zig.exe" cc -target x86-linux-gnu %*
