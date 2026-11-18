#include "kernel_parser.hpp"

#include <cctype>
#include <string>

namespace gpgpu::compiler {

namespace {

std::string_view trim(std::string_view sv) {
    size_t start = 0;
    while (start < sv.size() && std::isspace(static_cast<unsigned char>(sv[start]))) {
        ++start;
    }
    size_t end = sv.size();
    while (end > start && std::isspace(static_cast<unsigned char>(sv[end - 1]))) {
        --end;
    }
    return sv.substr(start, end - start);
}

} // namespace

bool KernelParser::parse(std::string_view source, ParsedKernel& outParsed, GPUError* outErr) {
    outParsed = ParsedKernel{};
    outParsed.rawSource = std::string(source);

    const std::string_view trimmed = trim(source);
    if (trimmed.empty()) {
        if (outErr) {
            *outErr = GPUError{GPUError::Code::KernelParseError, "Kernel source string is empty."};
        }
        return false;
    }

    // 1. Raw GLSL Mode: Starts with #version
    if (trimmed.starts_with("#version")) {
        outParsed.mode = KernelMode::RawGLSL;
        outParsed.bodyCode = std::string(trimmed);
        return true;
    }

    // 2. ShaderToy Mode: Contains void mainImage
    if (trimmed.find("mainImage") != std::string_view::npos) {
        outParsed.mode = KernelMode::ShaderToy;
        outParsed.bodyCode = std::string(trimmed);
        return true;
    }

    // 3. DSL Mode: kernel(coord) { ... }
    const size_t kernelPos = trimmed.find("kernel");
    if (kernelPos == std::string_view::npos) {
        if (outErr) {
            *outErr = GPUError{
                GPUError::Code::KernelParseError,
                "Expected 'kernel(coord) { ... }', 'mainImage(...)', or '#version 450' in kernel source."
            };
        }
        return false;
    }

    outParsed.mode = KernelMode::DSL;
    outParsed.preambleFunctions = std::string(trim(trimmed.substr(0, kernelPos)));

    const size_t openParen = trimmed.find('(', kernelPos + 6);
    const size_t closeParen = (openParen != std::string_view::npos) ? trimmed.find(')', openParen + 1) : std::string_view::npos;
    if (openParen == std::string_view::npos || closeParen == std::string_view::npos) {
        if (outErr) {
            *outErr = GPUError{GPUError::Code::KernelParseError, "Malformed 'kernel(coord)' parameter list."};
        }
        return false;
    }

    const std::string_view paramName = trim(trimmed.substr(openParen + 1, closeParen - openParen - 1));
    if (!paramName.empty()) {
        outParsed.coordParamName = std::string(paramName);
    } else {
        outParsed.coordParamName = "coord";
    }

    const size_t openBrace = trimmed.find('{', closeParen + 1);
    if (openBrace == std::string_view::npos) {
        if (outErr) {
            *outErr = GPUError{GPUError::Code::KernelParseError, "Missing '{' opening brace in kernel body."};
        }
        return false;
    }

    // Robust brace matching accounting for comments (// and /* */) and quotes
    int braceDepth = 0;
    size_t closeBrace = std::string_view::npos;
    bool inLineComment = false;
    bool inBlockComment = false;
    bool inString = false;

    for (size_t i = openBrace; i < trimmed.size(); ++i) {
        char c = trimmed[i];
        char next = (i + 1 < trimmed.size()) ? trimmed[i + 1] : '\0';

        if (inLineComment) {
            if (c == '\n') inLineComment = false;
            continue;
        }
        if (inBlockComment) {
            if (c == '*' && next == '/') {
                inBlockComment = false;
                ++i;
            }
            continue;
        }
        if (inString) {
            if (c == '\\') {
                ++i; // skip escaped character
            } else if (c == '"') {
                inString = false;
            }
            continue;
        }

        if (c == '/' && next == '/') {
            inLineComment = true;
            ++i;
            continue;
        }
        if (c == '/' && next == '*') {
            inBlockComment = true;
            ++i;
            continue;
        }
        if (c == '"') {
            inString = true;
            continue;
        }

        if (c == '{') {
            ++braceDepth;
        } else if (c == '}') {
            --braceDepth;
            if (braceDepth == 0) {
                closeBrace = i;
                break;
            }
        }
    }

    if (braceDepth != 0 || closeBrace == std::string_view::npos) {
        if (outErr) {
            *outErr = GPUError{GPUError::Code::KernelParseError, "Unbalanced or missing '}' closing brace in kernel body."};
        }
        return false;
    }

    std::string body(trim(trimmed.substr(openBrace + 1, closeBrace - openBrace - 1)));
    if (!body.empty() && body.back() != ';' && body.back() != '}') {
        body.push_back(';');
    }
    outParsed.bodyCode = std::move(body);
    return true;
}

} // namespace gpgpu::compiler
