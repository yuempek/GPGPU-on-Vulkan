# GPGPU.VULKAN — Sıfırdan İleri Seviyeye Öğretici Kullanım Kılavuzu (HOWTO)

Bu kılavuz, **GPGPU.VULKAN** kütüphanesini daha önce hiç Vulkan veya GPU programlama yapmamış bir geliştiricinin bile **5 dakika içinde** kendi projesine dahil edip en basitten en karmaşığa kadar kullanabilmesi için hazırlanmıştır.

Buradaki tüm kod parçacıkları (snippet'ler), projenin `tests/` klasöründeki 40 birim testinin doğruladığı tüm özellikleri öğretici ve yalın bir sırayla sunar.

---

## İçindekiler

1. [Kütüphane Klasör Yapısı (`deploy/`) ve Projeye Dahil Etme](#1-kütüphane-klasör-yapısı-deploy-ve-projeye-dahil-etme)
2. [Seviye 1: Merhaba GPU! (En Basit Kullanım Akışı)](#2-seviye-1-merhaba-gpu-en-basit-kullanım-akışı)
3. [Seviye 2: Cihaz Keşfi, GPU Seçimi ve Donanım Sorgulama](#3-seviye-2-cihaz-keşfi-gpu-seçimi-ve-donanım-sorgulama)
4. [Seviye 3: Bellek Yönetimi, Veri Tipleri ve `BufferPool`](#4-seviye-3-bellek-yönetimi-veri-tipleri-ve-bufferpool)
5. [Seviye 4: 1B (Dizi), 2B (Matris) ve 3B (Küp) Hesaplama](#5-seviye-4-1b-dizi-2b-matris-ve-3b-küp-hesaplama)
6. [Seviye 5: Çalışma Zamanında Dinamik Kernel Değiştirme ve Önbellek](#6-seviye-5-çalışma-zamanında-dinamik-kernel-değiştirme-ve-önbellek)
7. [Seviye 6: Ara Değer Hesaplama / Enterpolasyon (`nearest`, `linear`, `cubic`)](#7-seviye-6-ara-değer-hesaplama--enterpolasyon-nearest-linear-cubic)
8. [Seviye 7: ShaderToy Modu ve Ham (Raw) GLSL Modu](#8-seviye-7-shadertoy-modu-ve-ham-raw-glsl-modu)
9. [Seviye 8: Asenkron Çalıştırma (`runAsync`), `onDone` Callback ve Kernel Zincirleme](#9-seviye-8-asenkron-çalıştırma-runasync-ondone-callback-ve-kernel-zincirleme)
10. [Seviye 9: Hata Yakalama ve Shader Hata Ayıklama (`hasError`)](#10-seviye-9-hata-yakalama-ve-shader-hata-ayıklama-haserror)
11. [Seviye 10: C99, Python, MATLAB ve Java ile Kullanım](#11-seviye-10-c99-python-matlab-ve-java-ile-kullanım)

---

## 1. Kütüphane Klasör Yapısı (`deploy/`) ve Projeye Dahil Etme

Başka bir projede GPGPU.VULKAN'ı kullanmak için yalnızca bu `deploy/` klasörünü kopyalamanız yeterlidir. Başlık dosyaları **%100 saf C++20 / C99**'dur; sisteminizde Vulkan SDK kurulu olmasına dahi gerek yoktur.

```text
deploy/
├── HOWTO.md                        # Bu öğretici kılavuz
├── CMakeLists.txt                  # CMake projeleri için hazır entegrasyon dosyası
├── build_linux.sh                  # Linux ve Raspberry Pi derleme ve kurulum betiği
├── include/
│   └── gpgpu/
│       ├── gpgpu.hpp               # C++20 Ana Geliştirici Arayüzü (GPU, Data<T>, ExecutionTicket)
│       ├── gpgpu_c_api.h           # C99 Düz Arayüz (C, Python, MATLAB, Java, C# FFI için)
│       ├── types.hpp               # Veri tipleri, Dim3, GPUError, DeviceInfo, Capabilities
│       └── backend_interface.hpp   # Soyut donanım arayüzü
├── lib/
│   ├── gpgpu_vulkan.dll            # Dinamik çalışma zamanı kütüphanesi (Windows DLL)
│   ├── libgpgpu_vulkan.dll.a       # Dinamik bağlama (import) kütüphanesi (-lgpgpu_vulkan)
│   ├── libgpgpu_vulkan_static.a    # Tüm bağımlılıkları birleştirilmiş tek parça statik kütüphane
│   ├── libgpgpu_vulkan.so          # Linux paylaşımlı kütüphane (.so)
│   ├── windows-x64/                # Windows x86_64 ikilileri
│   ├── linux-x86_64/               # Linux PC Intel/AMD 64-bit paylaşımlı kütüphane (.so)
│   ├── linux-x86/                  # Linux PC Intel/AMD 32-bit paylaşımlı kütüphane (.so)
│   ├── linux-aarch64/              # Raspberry Pi 4/5 64-bit ARM (aarch64 V3DV) (.so)
│   └── linux-armhf/                # Raspberry Pi 2/3/4 32-bit ARM (armv7l) (.so)
└── bindings/
    ├── python/gpgpu.py             # Python sarmalayıcısı (NumPy + ctypes)
    ├── matlab/                     # MATLAB MEX ve sınıf sarmalayıcıları (GPGPU.m)
    └── java/                       # Java sınıfı (GPGPU.java) ve JNI köprüsü
```

### Yöntem A: CMake ile Kendi Projenize Bağlama (Önerilen)

Kendi projenizin `CMakeLists.txt` dosyasına şu 2 satırı ekleyin:

```cmake
cmake_minimum_required(VERSION 3.20)
project(BenimGPUProjem LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)

# 1. deploy klasörünü alt dizin olarak ekle
add_subdirectory(deploy)

add_executable(benim_uygulamam main.cpp)

# 2. İster statik (tek .exe, DLL gerektirmez) ister dinamik (shared) bağlayın:
target_link_libraries(benim_uygulamam PRIVATE gpgpu::static)
# veya dinamik DLL için: target_link_libraries(benim_uygulamam PRIVATE gpgpu::shared)
```

### Yöntem B: Komut Satırından Doğrudan Derleme (`g++`)

Hiç CMake kullanmadan tek komutla derlemek isterseniz:

```powershell
# Seçenek 1: Tek parça statik derleme (Çıkan .exe hiçbir DLL'e ihtiyaç duymaz!)
g++ main.cpp -std=c++20 -O3 -Ideploy/include -Ldeploy/lib -lgpgpu_vulkan_static -o app.exe

# Seçenek 2: Dinamik DLL ile hızlı derleme (gpgpu_vulkan.dll yanına kopyalanır)
g++ main.cpp -std=c++20 -O3 -Ideploy/include -Ldeploy/lib -lgpgpu_vulkan -o app.exe
```

---

## 2. Seviye 1: Merhaba GPU! (En Basit Kullanım Akışı)

GPU üzerinde hesaplama yapmanın mantığı 5 basit adımdan oluşur:

1. **`GPU` seç**
2. **Girdi ve çıktı verilerini tanımla (`createData`)**
3. **Yapılacak işlemi yaz (`kernel`)**
4. **Çalıştır (`run`)**
5. **Sonucu CPU'ya çek (`get`)**

```cpp
#include <gpgpu/gpgpu.hpp>
#include <iostream>
#include <vector>

int main() {
    using namespace gpgpu;

    // 1. Otomatik olarak sistemdeki en güçlü GPU'yu seç
    GPU gpu;

    // 2. CPU'da 5 elemanlı bir dizi hazırla ve GPU'ya gönder ("varNameIn" adıyla)
    std::vector<float> localInputData = {0.0f, 0.5f, 1.0f, 1.5707963f, 3.1415926f};
    auto remoteData = gpu.createData("varNameIn", localInputData);

    // 3. Çıktı değişkenini oluştur (Boyut vermenize gerek yok! run() sırasında otomatik ayrılır)
    auto remoteOutData = gpu.createData("outputData");

    // 4. GLSL Kernel kodunu tanımla (Sadece ana formülü yazın; noktalı virgül bile zorunlu değildir!)
    gpu.kernel("kernel(coord){ outputData[coord] = cos(varNameIn[coord]) }");

    // 5. 5 eleman için GPU'da çalıştır
    gpu.run(5);

    // 6. Sonucu GPU'dan CPU'ya geri çek
    std::vector<float> localOutputData = remoteOutData.get();

    for (size_t i = 0; i < localOutputData.size(); ++i) {
        std::cout << "cos(" << localInputData[i] << ") = " << localOutputData[i] << "\n";
    }

    // 7. Kaynakları temizle (GPU nesnesi kapsamdan çıkınca otomatik de temizlenir)
    gpu.dispose();
    return 0;
}
```

> **Neler Otomatik Yapıldı?**
> Siz sadece `outputData[coord] = cos(varNameIn[coord])` yazdınız. Kütüphane arka planda:
>
> - `#version 450` başlığını, `varNameIn` ve `outputData` için Vulkan Storage Buffer (SSBO) bağlantılarını oluşturdu.
> - `remoteOutData` için 5 adet `float` (`20 byte`) GPU belleğini tahsis etti.
> - Seçilen ekran kartına uygun çalışma grubu (workgroup) boyutunu ve taşma kontrolünü (`if (_gid.x >= 5) return;`) ekledi.
> - Kodu RAM içinde SPIR-V makine koduna derleyip çalıştırdı ve sonucu `localOutputData` vektörüne kopyaladı.

---

## 3. Seviye 2: Cihaz Keşfi, GPU Seçimi ve Donanım Sorgulama

*(Karşılık geldiği test: `tests/test_device_selection.cpp`)*

Bilgisayarınızda hem işlemciye gömülü bir GPU (örn. Intel/AMD iGPU), hem de harici bir ekran kartı (örn. NVIDIA RTX) olabilir. İstediğiniz kartı indeksle veya ismiyle seçebilirsiniz.

### 3.1. Sistemdeki Tüm GPU'ları Listeleme

```cpp
#include <gpgpu/gpgpu.hpp>
#include <iostream>

int main() {
    auto devices = gpgpu::GPU::devices();
    std::cout << "Bulunan Vulkan GPU sayisi: " << devices.size() << "\n";

    for (const auto& dev : devices) {
        std::cout << "[" << dev.index << "] " << dev.name
                  << " | Uretici: " << dev.vendor
                  << " | Bellek: " << (dev.totalMemoryBytes / (1024 * 1024)) << " MB"
                  << " | Paylasimli Bellek (UMA): " << (dev.isUnifiedMemory ? "Evet" : "Hayir")
                  << "\n";
    }
    return 0;
}
```

### 3.2. İndeks veya İsimle GPU Seçme

```cpp
using namespace gpgpu;

// Yöntem 1: Otomatik seçim (Harici GPU > Dahili GPU > Gömülü GPU sırasıyla puanlar)
GPU gpuAuto;

// Yöntem 2: İndeks ile seçim (0 numaralı kart)
GPU gpuByIndex(0);

// Yöntem 3: İsim parçasıyla seçim (Büyük/küçük harf duyarsızdır!)
GPU gpuByName("RTX 3060"); // veya GPU("Intel"), GPU("AMD"), GPU("V3D")
```

### 3.3. Seçilen Kartın Özelliklerini ve Bellek Durumunu Sorgulama

```cpp
GPU gpu(0);

DeviceInfo info = gpu.info();
Capabilities caps = gpu.capabilities();
MemoryInfo mem = gpu.memoryInfo();

std::cout << "Secilen Kart: " << info.name << " (Vulkan " << info.apiVersion << ")\n";
std::cout << "Max Workgroup Boyutu: " << caps.maxWorkGroupInvocations << " thread\n";
std::cout << "Subgroup (Warp/Wavefront) Boyutu: " << caps.subgroupSize << "\n";
std::cout << "Float64 (double) Destegi: " << (caps.supportsFloat64 ? "Var" : "Yok") << "\n";
std::cout << "Aktif GPU Tahsis Sayisi: " << mem.activeAllocations << "\n";
```

---

## 4. Seviye 3: Bellek Yönetimi, Veri Tipleri ve `BufferPool`

*(Karşılık geldiği test: `tests/test_memory_transfer.cpp`)*

### 4.1. Farklı Veri Tipleriyle Çalışma (`int32_t`, `uint8_t`, `vec4`, vb.)

`gpu.createData` şablonlu (templated) bir fonksiyondur. Veri verdiğinizde tipini otomatik anlar; boş oluştururken `<Tip>` belirterek istediğiniz tipte tampon açabilirsiniz:

```cpp
using namespace gpgpu;
GPU gpu;

// 1. Tam sayı (int32_t) dizisi
std::vector<int32_t> intInput = {-100, 0, 42, 1000};
auto intBuf = gpu.createData("intBuf", intInput); // Data<int32_t>

// 2. 4 bileşenli vektör (vec4 -> GLSL'de doğrudan vec4[] olarak görünür!)
std::vector<vec4> renkler = { {1.0f, 0.0f, 0.0f, 1.0f}, {0.0f, 1.0f, 0.0f, 1.0f} };
auto renkBuf = gpu.createData("renkBuf", renkler); // Data<vec4>

// 3. Ertelenmiş (Deferred) tipli çıktı oluşturma
auto ciktiInt  = gpu.createData<int32_t>("ciktiInt");
auto ciktiVec4 = gpu.createData<vec4>("ciktiVec4");
```

**Desteklenen C++ $\leftrightarrow$ GLSL Veri Tipleri:**

| C++ Tipi                 | GLSL Karşılığı                      | Boyut       |
|:------------------------ |:----------------------------------- |:----------- |
| `float` (Varsayılan)     | `float`                             | 4 Byte      |
| `double`                 | `double` (`GL_ARB_gpu_shader_fp64`) | 8 Byte      |
| `int32_t` (`int`)        | `int`                               | 4 Byte      |
| `uint32_t`               | `uint`                              | 4 Byte      |
| `int16_t` / `uint16_t`   | `int16_t` / `uint16_t`              | 2 Byte      |
| `int8_t` / `uint8_t`     | `int8_t` / `uint8_t`                | 1 Byte      |
| `gpgpu::vec2` / `vec4`   | `vec2` / `vec4`                     | 8 / 16 Byte |
| `gpgpu::ivec2` / `ivec4` | `ivec2` / `ivec4`                   | 8 / 16 Byte |

### 4.2. Mevcut Bir Tamponu Güncelleme (`set`) ve Sıfır Tahsisli Okuma (`getTo`)

Yeni bir `std::vector` kopyası oluşturmadan, önceden ayırdığınız belleğe doğrudan veri çekmek için `getTo` (veya `get(span)`) kullanabilirsiniz:

```cpp
GPU gpu;
std::vector<float> veri(1000, 1.0f);
auto remote = gpu.createData("veri", veri);

// Aynı GPU tamponunun içeriğini yeni verilerle güncelle (Yeni GPU belleği tahsis edilmez!)
veri[0] = 99.0f;
remote.set(veri);

// Sıfır heap tahsisi ile mevcut bir vektörün/dizinin içine doğrudan oku:
std::vector<float> mevcutTampon(1000);
remote.getTo(mevcutTampon); // Yeni bellek ayırmaz, doğrudan mevcutTampon içine yazar!
```

### 4.3. Otomatik Bellek Havuzu (`BufferPool`)

Bir `Data` nesnesini `dispose()` ettiğinizde veya yeniden boyutlandırdığınızda, Vulkan tamponu hemen sürücüye iade edilmez; kütüphanenin iç `BufferPool` havuzuna alınır. Aynı (veya yakın) boyutta yeni bir veri oluşturduğunuzda **0 nanosaniye** Vulkan tahsis maliyetiyle havuzdan verilir:

```cpp
auto d1 = gpu.createData("buf1", std::vector<float>(100000, 5.0f));
d1.dispose(); // Tampon havuza (pool) taşındı: pooledAllocations == 1

// Aynı boyutta yeni tampon istendiğinde sürücüye gitmeden havuzdan anında alınır!
auto d2 = gpu.createData("buf2", std::vector<float>(100000, 10.0f));
```

---

## 5. Seviye 4: 1B (Dizi), 2B (Matris) ve 3B (Küp) Hesaplama

*(Karşılık geldiği testler: `tests/test_compute_raw.cpp` ve `tests/test_kernel_dsl.cpp`)*

Normal Vulkan'da çalışma grubu (workgroup) sayılarını hesaplamak ve 2B/3B koordinatları tek boyutlu dizi indeksine (`(z * H + y) * W + x`) çevirmek geliştiricinin yüküdür. **GPGPU.VULKAN bunu tamamen otomatikleştirir.**

`gpu.run(...)` fonksiyonuna kaç parametre verirseniz, kernel içindeki `coord` değişkeni otomatik olarak o boyuta bürünür:

| Çağrı Şekli              | Çalışma Modu          | Kernel'deki `coord` Tipi | Erişim Örneği                              |
|:------------------------ |:--------------------- |:------------------------ |:------------------------------------------ |
| `gpu.run(1000)`          | 1B (Dizi)             | `int`                    | `coord` (`0 .. 999`)                       |
| `gpu.run(800, 600)`      | 2B (Matris / Görüntü) | `ivec2`                  | `coord.x` (`0..799`), `coord.y` (`0..599`) |
| `gpu.run(100, 100, 100)` | 3B (Küp / Hacim)      | `ivec3`                  | `coord.x`, `coord.y`, `coord.z` (`0..99`)  |

### 5.1. 1B Örnek: 1 Milyon Elemanlı Dizi İşleme

```cpp
GPU gpu;
const uint32_t N = 1'000'000;

std::vector<float> girdi(N, 0.5f);
auto inData  = gpu.createData("inData", girdi);
auto outData = gpu.createData("outData");

// coord burada 'int' tipindedir
gpu.kernel("kernel(coord){ outData[coord] = cos(inData[coord]) + sin(inData[coord]) }");
gpu.run(N);

auto sonuc = outData.get(); // 1.000.000 elemanlı sonuç vektörü
```

### 5.2. 2B Örnek: `800 x 600` Matris / Görüntü İşleme

16'ya tam bölünmeyen boyutlar (örneğin `600 / 16 = 37.5`) verseniz bile kütüphane taşma kontrolünü otomatik ekler; asla bellek dışına yazılmaz:

```cpp
GPU gpu;
const uint32_t width = 800, height = 600;

auto outMatrix = gpu.createData("outMatrix");

// gpu.run(width, height) çağrıldığında coord otomatik olarak 'ivec2' olur!
// outMatrix[coord] yazdığınızda arka planda outMatrix[coord.y * 800 + coord.x] adresine erişilir.
gpu.kernel(R"(
    kernel(coord) {
        outMatrix[coord] = float(coord.x) + float(coord.y) * 10.0;
    }
)");

gpu.run(width, height); // 800x600 matris olarak çalıştır

auto hostMatrix = outMatrix.get();
// (x=5, y=2) noktasının değeri: 5 + 2*10 = 25.0
float deger = hostMatrix[2 * width + 5];
```

### 5.3. 3B Örnek: `100 x 100 x 100` Küp (Voxel) İşleme

```cpp
GPU gpu;
auto outCube = gpu.createData("outCube");

// gpu.run(100, 100, 100) çağrıldığında coord otomatik olarak 'ivec3' olur!
gpu.kernel(R"(
    kernel(coord) {
        outCube[coord] = float(coord.x) + float(coord.y) * 100.0 + float(coord.z) * 10000.0;
    }
)");

gpu.run(100, 100, 100); // 1.000.000 elemanlı 3B küp

auto hostCube = outCube.get();
```

---

## 6. Seviye 5: Çalışma Zamanında Dinamik Kernel Değiştirme ve Önbellek

*(Karşılık geldiği test: `tests/test_kernel_dsl.cpp`)*

Uygulamanız çalışırken GPU'daki verileri hiç silmeden kernel kodunu istediğiniz kadar değiştirebilirsiniz.

```cpp
GPU gpu;
std::vector<float> girdi(100, 1.0f);
auto inData  = gpu.createData("inData", girdi);
auto outData = gpu.createData("outData");

// 1. Önce Kosinüs hesapla
gpu.kernel("kernel(coord){ outData[coord] = cos(inData[coord]) }");
gpu.run(100);
auto cosSonuc = outData.get();

// 2. Çalışma zamanında kernel'i Sinüs olarak değiştir ve aynı verilerle tekrar çalıştır!
gpu.kernel("kernel(coord){ outData[coord] = sin(inData[coord]) }");
gpu.run(100);
auto sinSonuc = outData.get();
```

> **Sıfır Tekrar Derleme Maliyeti (L1 RAM & L2 Disk Önbelleği):**
> Bir döngü içinde aynı kernel'i 1.000 kez çalıştırdığınızda (`for(int i=0; i<1000; ++i) gpu.run(100);`), GLSL derleyicisi **yalnızca ilk çalışmada (1 kez)** devreye girer. Sonraki 999 çalıştırma, `XXH3_64` önbelleğinden **~20 nanosaniye** içinde hazır `VkPipeline` olarak çekilir. Ayrıca derlenen SPIR-V kodları diske (`%LOCALAPPDATA%/gpgpu_vulkan/cache/`) kaydedildiği için programı kapatıp açtığınızda bile anında yüklenir!

---

## 7. Seviye 6: Ara Değer Hesaplama / Enterpolasyon (`nearest`, `linear`, `cubic`)

*(Karşılık geldiği test: `tests/test_interpolation.cpp`)*

Görüntü işleme, sinyal işleme veya bilimsel simülasyonlarda bir dizinin/matrisin sadece tam sayı indekslerini (`inData[1]`, `inData[2]`) değil, **ara noktalarını** (`inData[1.5]` veya `inGrid[vec2(0.5, 0.5)]`) okumak isteyebilirsiniz.

Bunun için `createData` çağrısında son parametre olarak `"nearest"`, `"linear"` (veya `"lineer"`) ya da `"cubic"` vermeniz yeterlidir!

### 7.1. En Yakın Komşu (`"nearest"`)

Kayan noktalı koordinatı en yakın tam sayı indekse yuvarlar:

```cpp
GPU gpu;
std::vector<float> dizi = {0.0f, 10.0f, 20.0f, 30.0f};

// "nearest" enterpolasyon modunda oluştur
auto inData  = gpu.createData("inData", dizi, "nearest");
auto outData = gpu.createData("outData");

gpu.kernel(R"(
    kernel(coord) {
        if (coord == 0) outData[coord] = inData[0.2];      // 0'a yuvarlanır -> 0.0
        else if (coord == 1) outData[coord] = inData[0.8]; // 1'e yuvarlanır -> 10.0
        else if (coord == 2) outData[coord] = inData[1.4]; // 1'e yuvarlanır -> 10.0
        else outData[coord] = inData[2.7];                 // 3'e yuvarlanır -> 30.0
    }
)");
gpu.run(4);
```

### 7.2. Çift Doğrusal / Bilineer Enterpolasyon (`"linear"` / `"lineer"`)

2B bir matriste `vec2(x, y)` ile ara nokta okuduğunuzda 4 komşu pikseli ağırlıklı olarak karıştırır:

```cpp
GPU gpu;
// 2x2 Matris:
// (0,0)=0.0,  (1,0)=10.0
// (0,1)=20.0, (1,1)=30.0
std::vector<float> matris2x2 = { 0.0f, 10.0f, 20.0f, 30.0f };

// Boyutu Dim3(2, 2) ve enterpolasyonu "linear" olarak belirtiyoruz
auto inGrid = gpu.createData("inGrid", matris2x2, Dim3(2, 2), "linear");
auto outVal = gpu.createData("outVal");

// Tam orta noktayı (0.5, 0.5) oku -> 4 köşenin ortalaması: (0 + 10 + 20 + 30) / 4 = 15.0
gpu.kernel("kernel(coord){ outVal[coord] = inGrid[vec2(0.5, 0.5)] }");
gpu.run(1);

std::cout << "Merkez deger: " << outVal.get()[0] << "\n"; // 15.0 yazdırır!
```

### 7.3. Bikübik (Catmull-Rom Spline) Enterpolasyon (`"cubic"`)

Görüntü büyütme (upsampling) ve pürüzsüz eğri uydurma için 1B'de 4 komşu noktayı, 2B'de $4 \times 4 = 16$ komşu noktayı Catmull-Rom kübik ağırlıklarıyla hesaplar:

```cpp
GPU gpu;
// 4x4 giriş görüntüsü
std::vector<float> kucukGoruntu = {
     0.0f, 10.0f, 20.0f, 30.0f,
    10.0f, 20.0f, 30.0f, 40.0f,
    20.0f, 30.0f, 40.0f, 50.0f,
    30.0f, 40.0f, 50.0f, 60.0f
};

auto srcImg = gpu.createData("srcImg", kucukGoruntu, Dim3(4, 4), "cubic");
auto outImg = gpu.createData("outImg");

// Görüntüyü 2 kat (8x8) bikübik olarak büyüt
gpu.kernel(R"(
    kernel(coord) {
        vec2 uv = vec2(coord) * 0.5;
        outImg[coord] = srcImg[uv];
    }
)");
gpu.run(8, 8);
auto buyukGoruntu = outImg.get(); // 64 elemanlı pürüzsüz büyütülmüş matris
```

---

## 8. Seviye 7: ShaderToy Modu ve Ham (Raw) GLSL Modu

*(Karşılık geldiği testler: `tests/test_kernel_dsl.cpp` ve `tests/test_compute_raw.cpp`)*

`gpu.kernel(...)` fonksiyonu verdiğiniz kodu analiz ederek **3 farklı modu** otomatik tanır:

1. **`kernel(coord) { ... }`** $\rightarrow$ Basit DSL Modu (Yukarıda gördüğümüz mod)
2. **`void mainImage(out vec4 fragColor, in vec2 fragCoord)`** $\rightarrow$ ShaderToy Modu
3. **`#version 450 ...`** $\rightarrow$ Ham (Raw) GLSL Compute Shader Modu

### 8.1. ShaderToy Modu (`mainImage`)

ShaderToy sitesindeki bir piksel shader kodunu kopyalayıp doğrudan `gpu.kernel(...)` içine yapıştırabilirsiniz! `iResolution`, `iTime`, `iTimeDelta`, `iFrame` ve `iMouse` değişkenleri otomatik olarak tanımlanır:

```cpp
GPU gpu;
const uint32_t W = 320, H = 240;

// RGBA pikselleri için vec4 tipinde çıktı tamponu
auto outputData = gpu.createData<vec4>("outputData");

gpu.kernel(R"(
    void mainImage(out vec4 fragColor, in vec2 fragCoord) {
        vec2 uv = fragCoord / iResolution.xy;
        vec3 col = 0.5 + 0.5 * cos(iTime + uv.xyx + vec3(0.0, 2.0, 4.0));
        fragColor = vec4(col, 1.0);
    }
)");

gpu.run(W, H);
std::vector<vec4> pixels = outputData.get(); // 320x240 RGBA float piksel dizisi!
```

### 8.2. Yardımcı Fonksiyonlu DSL veya Ham `#version 450` GLSL Modu

DSL modunda `kernel(coord)` bloğunun üstüne kendi GLSL yardımcı fonksiyonlarınızı da yazabilirsiniz:

```cpp
gpu.kernel(R"(
    // Kullanıcı tanımlı yardımcı GLSL fonksiyonu
    float kareAlVeEkle(float x, float ek) {
        return x * x + ek;
    }

    kernel(coord) {
        outData[coord] = kareAlVeEkle(inData[coord], 5.0);
    }
)");
```

Veya `#version 450` ile başlayarak tüm shader'ı sıfırdan kendiniz yazabilirsiniz:

```cpp
gpu.kernel(R"(
    #version 450
    layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
    layout(push_constant) uniform PushConstants { uvec4 dims; } pc;
    layout(set = 0, binding = 0, std430) readonly buffer B0 { float inData[]; };
    layout(set = 0, binding = 1, std430) writeonly buffer B1 { float outData[]; };

    void main() {
        uint idx = gl_GlobalInvocationID.x;
        if (idx >= pc.dims.x) return;
        outData[idx] = cos(inData[idx]);
    }
)");
```

---

## 9. Seviye 8: Asenkron Çalıştırma (`runAsync`), `onDone` Callback ve Kernel Zincirleme

*(Karşılık geldiği test: `tests/test_compute_raw.cpp` ve `examples/04_async_chaining.cpp`)*

`gpu.run(...)` senkrondur; yani GPU işi bitirene kadar CPU'yu bekletir (`runAsync + wait`).
Büyük hesaplamalarda GPU çalışırken CPU'nun başka işler yapmaya devam etmesi için **`runAsync`** kullanılır.

### 9.1. `runAsync` + `wait` ve `onDone` Callback Kullanımı

```cpp
#include <gpgpu/gpgpu.hpp>
#include <atomic>
#include <iostream>

int main() {
    using namespace gpgpu;
    GPU gpu;

    const uint32_t N = 100'000;
    auto inData  = gpu.createData("inData", std::vector<float>(N, 2.0f));
    auto outData = gpu.createData("outData");

    gpu.kernel("kernel(coord){ outData[coord] = inData[coord] * inData[coord] }");

    std::atomic<bool> bittiMi{false};

    // 1. İşi GPU'ya asenkron gönder (CPU hiç beklemeden alt satıra geçer!)
    ExecutionTicket ticket = gpu.runAsync(N);

    // 2. GPU işi bitirdiğinde arka planda otomatik çağrılacak fonksiyonu kaydet
    ticket.onDone([&]() {
        std::cout << "[Callback] GPU hesaplamayi tamamladi!\n";
        bittiMi.store(true);
    });

    // ... Burada CPU başka işler yapabilir ...

    // 3. Sonucu okumadan önce GPU'nun bitmesini bekle
    ticket.wait(); // (veya gpu.wait(); ya da doğrudan outData.get() çağırabilirsiniz, get() otomatik bekler)

    std::cout << "Donanimsal GPU suresi: " << ticket.elapsedMilliseconds() << " ms\n";
    auto sonuc = outData.get();
    return 0;
}
```

### 9.2. Sıfır Kopyalamalı Kernel Zincirleme (Kernel Chaining)

Arka arkaya birden fazla kernel çalıştırdığınızda, ara sonuçları CPU'ya çekmenize (`get()`) gerek yoktur. Veriler doğrudan ekran kartı belleğinde (VRAM) kalır ve kütüphane iki kernel arasına otomatik olarak Vulkan bellek bariyeri (`VkMemoryBarrier`) koyar:

```cpp
GPU gpu;
const uint32_t N = 1'000'000;

auto bufA = gpu.createData("bufA", std::vector<float>(N, 1.0f));
auto bufB = gpu.createData("bufB"); // Ara tampon (Sadece GPU'da yaşar, CPU'ya hiç kopyalanmaz!)
auto bufC = gpu.createData("bufC"); // Nihai çıktı tamponu

// Adım 1: bufB = bufA * 2.0
gpu.kernel("kernel(coord){ bufB[coord] = bufA[coord] * 2.0 }");
gpu.runAsync(N);

// Adım 2: bufC = sqrt(bufB) + 1.0 (bufB doğrudan VRAM üzerinden okunur!)
gpu.kernel("kernel(coord){ bufC[coord] = sqrt(bufB[coord]) + 1.0 }");
gpu.run(N);

// Sadece en son sonucu CPU'ya çekiyoruz!
auto nihaiSonuc = bufC.get();
```

---

## 10. Seviye 9: Hata Yakalama ve Shader Hata Ayıklama (`hasError`)

*(Karşılık geldiği testler: `tests/test_device_selection.cpp`, `tests/test_compute_raw.cpp`, `tests/test_c_api.cpp`)*

Kütüphane hatalı bir GPU adı, bozuk bir GLSL kodu veya geçersiz bir boyut verildiğinde **asla çökmez (exception fırlatıp programı sonlandırmaz)**. Bunun yerine hatayı kaydeder ve `gpu.hasError()` üzerinden size sunar:

```cpp
GPU gpu;

// Bilerek bozuk bir GLSL kodu verelim ('tanimsiz_degisken' diye bir şey yok)
gpu.kernel("kernel(coord){ outData[coord] = tanimsiz_degisken * 2.0; }");
gpu.run(100);

// Hata kontrolü: Hata yoksa std::nullopt (false), hata varsa GPUError döner
if (auto err = gpu.hasError()) {
    std::cout << "Hata Yakalandi!\n";
    std::cout << "  Mesaj      : " << err->message << "\n";
    std::cout << "  Shader Log : " << err->shaderLog << "\n";
    std::cout << "  Uretilen GLSL:\n" << err->generatedGlsl << "\n";

    // Hatayı temizleyip normal çalışmaya devam edebiliriz
    gpu.clearError();
}
```

---

## 11. Seviye 10: C99, Python, MATLAB ve Java ile Kullanım

*(Karşılık geldiği testler: `tests/test_c_api.cpp` ve `tests/test_python_binding.py`)*

### 11.1. Saf C (C99) ile Kullanım (`include/gpgpu/gpgpu_c_api.h`)

C++ şablonları kullanmayan düz C projeleri veya diğer dillerin FFI bağlantıları için:

```c
#include <gpgpu/gpgpu_c_api.h>
#include <stdio.h>

int main(void) {
    // 1. Cihaz oluştur
    GPGPU_Context ctx = gpgpu_create_auto();

    // 2. Girdi ve çıktı verilerini tanımla
    float input[4] = {0.0f, 0.5f, 1.0f, 1.5707963f};
    GPGPU_Data inHandle  = gpgpu_create_data(ctx, "varNameIn", input, 4, GPGPU_DTYPE_FLOAT32, GPGPU_INTERP_NONE);
    GPGPU_Data outHandle = gpgpu_create_data(ctx, "outputData", NULL, 0, GPGPU_DTYPE_FLOAT32, GPGPU_INTERP_NONE);

    // 3. Kernel tanımla ve çalıştır
    gpgpu_set_kernel(ctx, "kernel(coord){ outputData[coord] = cos(varNameIn[coord]); }");
    gpgpu_run(ctx, 4, 1, 1);

    // 4. Hata kontrolü (Hata yoksa NULL döner)
    const char* err = gpgpu_has_error(ctx);
    if (err != NULL) {
        printf("Hata: %s\n", err);
    }

    // 5. Sonucu oku
    float output[4] = {0};
    gpgpu_data_get(outHandle, output, 4);
    for (int i = 0; i < 4; ++i) {
        printf("cos(%f) = %f\n", input[i], output[i]);
    }

    // 6. Temizle
    gpgpu_data_destroy(inHandle);
    gpgpu_data_destroy(outHandle);
    gpgpu_destroy(ctx);
    return 0;
}
```

### 11.2. Python ile Kullanım (`deploy/bindings/python/gpgpu.py`)

Hiçbir `pip` paketi veya C derleyicisi gerektirmez (`numpy` yüklüyse doğrudan `ndarray` döndürür, yüklü değilse standart Python `list` ile çalışır):

```python
import sys
sys.path.append("deploy/bindings/python")
from gpgpu import GPU

# Cihazları listele
print("Cihazlar:", GPU.devices())

# Context Manager (with) ile otomatik temizlenen kullanım:
with GPU(0) as gpu:
    girdi = [0.0, 0.5, 1.0, 1.5707963]
    remote_in  = gpu.createData("varNameIn", girdi)
    remote_out = gpu.createData("outputData")

    gpu.kernel("kernel(coord){ outputData[coord] = cos(varNameIn[coord]) }")
    gpu.run(len(girdi))

    sonuc = remote_out.get()
    print("Sonuc:", sonuc)
    print(f"GPU Suresi: {gpu.lastKernelTimeMs():.3f} ms")
```

#### 11.2.1. Linux ve Raspberry Pi (ARM) Üzerinde Çalıştırma

> [!IMPORTANT]
> **Mimari Uyumluluğu ve Önceden Derlenmiş `.so` Dosyaları:**
> 1. `deploy/lib/` dizininde tüm hedef mimariler için yerel `.so` ve `.dll` dosyaları **önceden derlenmiş (pre-compiled)** olarak hazır bulunmaktadır:
>    - **Raspberry Pi 4/5 (64-bit OS):** `deploy/lib/linux-aarch64/libgpgpu_vulkan.so`
>    - **Raspberry Pi 2/3/4 (32-bit OS):** `deploy/lib/linux-armhf/libgpgpu_vulkan.so`
>    - **Linux PC (64-bit Intel/AMD):** `deploy/lib/linux-x86_64/libgpgpu_vulkan.so`
>    - **Linux PC (32-bit Intel/AMD):** `deploy/lib/linux-x86/libgpgpu_vulkan.so`
>    - **Windows PC (64-bit):** `deploy/lib/windows-x64/gpgpu_vulkan.dll`
> 2. `gpgpu.py` sarmalayıcısı çalıştırıldığı işletim sistemini ve CPU mimarisini otomatik algılayarak doğru `.so` dosyasını seçer. Bu sayede Raspberry Pi'nize yalnızca `deploy/` klasörünü kopyalayıp doğrudan `python3 main.py` komutuyla çalıştırabilirsiniz!
> 3. Eğer kendiniz yeniden derlemek isterseniz:
>    - **Windows üzerinden Cross-Compile (tüm mimariler için):** `powershell .\build_cross_linux.ps1`
>    - **Raspberry Pi / Linux üzerinde doğrudan yerel derleme:** `./build.sh` (veya `bash deploy/build_linux.sh`) (CMake 3.18+ ile %100 uyumludur)

**Raspberry Pi / Linux Üzerinde Tek Komutla Derleme:**
```bash
# Gerekli paketleri kurun (Debian / Ubuntu / Raspberry Pi OS):
sudo apt update && sudo apt install -y build-essential cmake ninja-build libvulkan-dev mesa-vulkan-drivers

# Proje ana dizininden derleyip deploy klasörünü güncellemek için:
./build.sh

# Veya sadece deploy klasörü varsa:
bash deploy/build_linux.sh
```

Derleme tamamlandığında `libgpgpu_vulkan.so` oluşturulur ve Python kodu Raspberry Pi'nizin VideoCore GPU'su (V3DV) üzerinde doğrudan çalışır!

### 11.3. MATLAB ile Kullanım (`deploy/bindings/matlab/GPGPU.m`)

MATLAB komut penceresinde bir kez `mex -Ideploy/include -Ldeploy/lib -lgpgpu_vulkan.dll deploy/bindings/matlab/gpgpu_mex.cpp` çalıştırdıktan sonra:

```matlab
% MATLAB komut penceresinde bir kez 
% addpath('deploy/lib');
% addpath('deploy/bindings/matlab');
% setenv('MW_MINGW64_LOC','C:\msys64\mingw64')
% clear mex
% mex -setup C++
% mex -Ideploy/include -Ldeploy/lib -lgpgpu_vulkan.dll deploy/bindings/matlab/gpgpu_mex.cpp
% projectDir = 'C:\Users\X\Desktop\_PROJECTS\GPU\GPGPU.VULKAN.MATLAB.TEST';
% dllDir = fullfile(projectDir, 'deploy', 'lib');
% setenv('PATH', [dllDir ';' getenv('PATH')]);addpath('deploy/bindings/matlab');

gpu = GPGPU(0);
A = single( linspace(0, pi, 1000) );

remoteIn  = gpu.createData('varNameIn', A);
remoteOut = gpu.createData('outputData');

gpu.kernel('kernel(coord){ outputData[coord] = cos(varNameIn[coord]); }');
gpu.run(1000);

B = remoteOut.get();
gpu.dispose();
```

### 11.4. Java ile Kullanım (`deploy/bindings/java/GPGPU.java`)

```java
import com.gpgpu.GPGPU;

public class Main {
    public static void main(String[] args) {
        try (GPGPU gpu = new GPGPU(0)) {
            float[] input = new float[]{0.0f, 0.5f, 1.0f, 1.5707963f};
            GPGPU.Data inData  = gpu.createData("varNameIn", input);
            GPGPU.Data outData = gpu.createData("outputData");

            gpu.kernel("kernel(coord){ outputData[coord] = cos(varNameIn[coord]); }");
            gpu.run(input.length);

            float[] output = outData.get();
            for (float v : output) {
                System.out.println(v);
            }
        }
    }
}
```
