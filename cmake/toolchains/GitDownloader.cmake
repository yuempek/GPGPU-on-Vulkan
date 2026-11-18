# ==============================================================================
# GitDownloader.cmake
# SADECE KAYNAK İNDİRİR. Hiçbir bağımlılık configure/derleme edilmez,
# projeye add_subdirectory() ile dahil edilmez.
#
# Her bağımlılık şu klasöre indirilir: RootPath/<Ad>
#
# Bu klasörleri daha sonra başka bir CMake projesinden kendiniz
# add_subdirectory(...) veya target_include_directories(...) ile bağlayabilirsiniz.
#
# Kullanım:
#   include(${CMAKE_CURRENT_SOURCE_DIR}/GitDownloader.cmake)
#   set(MY_DEPS "ad|https://...git|tag-veya-commit" ...)
#   GitDownloader("${CMAKE_CURRENT_SOURCE_DIR}/external" "${MY_DEPS}")
# ==============================================================================


# --- git bul ---------------------------------------------------------------
find_package(Git QUIET)
if(NOT GIT_EXECUTABLE)
    find_program(GIT_EXECUTABLE git)
endif()
if(NOT GIT_EXECUTABLE)
    message(FATAL_ERROR "git bulunamadi. Lutfen git'i PATH'e ekleyin.")
endif()


# --- İndirme fonksiyonu ----------------------------------------------------
# git_downloader_worker(<RootPath> <ad> <git-url> <tag|branch|commit>)
# Sonucu GIT_DOWNLOADER_RESULT degiskeni ile cagirana bildirir: "yeni" | "atlandi"
function(git_downloader_worker root name url ref)
    # --- Girdi doğrulama ---------------------------------------------------
    if("${name}" STREQUAL "" OR "${url}" STREQUAL "" OR "${ref}" STREQUAL "")
        message(FATAL_ERROR
            "[GitDownloader] Eksik tanim: ad='${name}' url='${url}' ref='${ref}'")
    endif()
    # Klasör adı yol ayırıcı içeremez (yanlışlıkla RootPath'in içine yazılmasın)
    if(name MATCHES "[/\\\\]")
        message(FATAL_ERROR "[GitDownloader] Gecersiz klasor adi: '${name}'")
    endif()

    # RootPath her zaman mutlak ve mevcut olmalı
    get_filename_component(root_abs "${root}" ABSOLUTE)
    file(MAKE_DIRECTORY "${root_abs}")

    set(dest "${root_abs}/${name}")

    # Zaten indirilmişse atla
    if(EXISTS "${dest}/.git")
        execute_process(
            COMMAND "${GIT_EXECUTABLE}" -C "${dest}" rev-parse --verify HEAD
            RESULT_VARIABLE have_head
            OUTPUT_QUIET ERROR_QUIET)
        if(have_head EQUAL 0)
            message(STATUS "[GitDownloader] ${name} : zaten mevcut, atlandi")
            set(GIT_DOWNLOADER_RESULT "atlandi" PARENT_SCOPE)
            return()
        endif()
        message(STATUS "[GitDownloader] ${name} : bozuk klon bulundu, yeniden indiriliyor")
    endif()

    message(STATUS "[GitDownloader] ${name} indiriliyor : ${url} @ ${ref}")

    file(REMOVE_RECURSE "${dest}")
    file(MAKE_DIRECTORY "${dest}")

    set(fetch_err "")
    set(checkout_err "")

    execute_process(
        COMMAND "${GIT_EXECUTABLE}" init -q "${dest}"
        RESULT_VARIABLE rc
        OUTPUT_QUIET ERROR_VARIABLE init_err)
    if(NOT rc EQUAL 0)
        file(REMOVE_RECURSE "${dest}")
        message(FATAL_ERROR "[GitDownloader] ${name} : git init basarisiz\n${init_err}")
    endif()

    execute_process(
        COMMAND "${GIT_EXECUTABLE}" -C "${dest}" remote add origin "${url}"
        RESULT_VARIABLE rc
        OUTPUT_QUIET ERROR_VARIABLE remote_err)
    if(NOT rc EQUAL 0)
        file(REMOVE_RECURSE "${dest}")
        message(FATAL_ERROR "[GitDownloader] ${name} : git remote add basarisiz\n${remote_err}")
    endif()

    # 1. deneme: tag / branch adini dogrudan sig cek
    execute_process(
        COMMAND "${GIT_EXECUTABLE}" -C "${dest}" -c advice.detachedHead=false
                fetch --depth 1 --no-tags origin "${ref}"
        RESULT_VARIABLE rc
        OUTPUT_QUIET ERROR_VARIABLE fetch_err)
    if(rc EQUAL 0)
        execute_process(
            COMMAND "${GIT_EXECUTABLE}" -C "${dest}" checkout -q FETCH_HEAD
            RESULT_VARIABLE rc
            OUTPUT_QUIET ERROR_VARIABLE checkout_err)
    else()
        # 2. deneme: commit hash'leri icin once varsayilan dali sig cek
        execute_process(
            COMMAND "${GIT_EXECUTABLE}" -C "${dest}" -c advice.detachedHead=false
                    fetch --depth 1 --no-tags origin HEAD
            RESULT_VARIABLE rc
            OUTPUT_QUIET ERROR_VARIABLE fetch_err)
        if(rc EQUAL 0)
            execute_process(
                COMMAND "${GIT_EXECUTABLE}" -C "${dest}" checkout -q "${ref}"
                RESULT_VARIABLE rc
                OUTPUT_QUIET ERROR_VARIABLE checkout_err)
        endif()
        # 3. deneme: tum gecmisi cek
        if(NOT rc EQUAL 0)
            execute_process(
                COMMAND "${GIT_EXECUTABLE}" -C "${dest}" fetch --tags origin
                RESULT_VARIABLE rc
                OUTPUT_QUIET ERROR_VARIABLE fetch_err)
            if(rc EQUAL 0)
                execute_process(
                    COMMAND "${GIT_EXECUTABLE}" -C "${dest}" checkout -q "${ref}"
                    RESULT_VARIABLE rc
                    OUTPUT_QUIET ERROR_VARIABLE checkout_err)
            endif()
        endif()
    endif()

    # Checkout gercekten basarili mi? (bos calisma agaci birakma)
    if(rc EQUAL 0)
        execute_process(
            COMMAND "${GIT_EXECUTABLE}" -C "${dest}" rev-parse --verify HEAD
            RESULT_VARIABLE rc
            OUTPUT_QUIET ERROR_QUIET)
    endif()

    if(NOT rc EQUAL 0)
        file(REMOVE_RECURSE "${dest}")
        message(FATAL_ERROR
            "[GitDownloader] ${name} indirilemedi (${ref}).\n"
            "git: ${fetch_err}\n${checkout_err}")
    endif()

    execute_process(
        COMMAND "${GIT_EXECUTABLE}" -C "${dest}" rev-parse --short HEAD
        OUTPUT_VARIABLE head OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET)
    message(STATUS "[GitDownloader] ${name} hazir (commit ${head})")

    set(GIT_DOWNLOADER_RESULT "yeni" PARENT_SCOPE)
endfunction()


# --- Toplu indirme ---------------------------------------------------------
# GitDownloader(<RootPath> <NAME_URL_REF listesi>)
# Liste bicimi: "<ad>|<git url>|<tag / branch / commit>"
function(GitDownloader RootPath NAME_URL_REF)
    get_filename_component(root_abs "${RootPath}" ABSOLUTE)
    file(MAKE_DIRECTORY "${root_abs}")

    set(count_new 0)
    set(count_skip 0)

    foreach(entry IN LISTS NAME_URL_REF)
        string(STRIP "${entry}" entry)

        # Bos satirlari ve yorumlari atla
        if(entry STREQUAL "" OR entry MATCHES "^#")
            continue()
        endif()

        string(REPLACE "|" ";" parts "${entry}")
        list(LENGTH parts parts_len)
        if(NOT parts_len EQUAL 3)
            message(FATAL_ERROR
                "[GitDownloader] Gecersiz tanim: '${entry}'\n"
                "Beklenen bicim: <ad>|<git-url>|<tag/branch/commit>")
        endif()

        list(GET parts 0 dep_name)
        list(GET parts 1 dep_url)
        list(GET parts 2 dep_ref)
        string(STRIP "${dep_name}" dep_name)
        string(STRIP "${dep_url}"  dep_url)
        string(STRIP "${dep_ref}"  dep_ref)

        git_downloader_worker("${root_abs}" "${dep_name}" "${dep_url}" "${dep_ref}")

        if(GIT_DOWNLOADER_RESULT STREQUAL "atlandi")
            math(EXPR count_skip "${count_skip} + 1")
        else()
            math(EXPR count_new "${count_new} + 1")
        endif()
    endforeach()

    message(STATUS "[GitDownloader] Ozet: ${count_new} yeni indirildi, ${count_skip} atlandi -> ${root_abs}")
endfunction()
