@echo off
"%~dp0..\zig\zig.exe" c++ -target x86-linux-gnu %*
