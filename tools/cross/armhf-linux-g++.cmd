@echo off
"%~dp0..\zig\zig.exe" c++ -target arm-linux-gnueabihf %*
