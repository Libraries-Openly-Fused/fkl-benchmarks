// Where does the time of the HIP graph of the horizontal fusion benchmark go? The same per-crop kernel (crop + saturate + mul + sub + div + write, 60x120 uchar -> float of a
// 3840x2160 image) launched four ways, for BATCH crops:
//   plain        BATCH launches in the stream (the "iterated batch" of the benchmark)
//   children     the graph of the benchmark: BATCH child graph nodes, each captured from one launch (hipStreamBeginCapture, global mode), no dependencies between them
//   children_seq the same with every child depending on the previous one
//   flat         ONE stream capture of the BATCH launches: a graph of BATCH kernel nodes in a chain
// per variant: submit = time until the launch call(s) return (host), total = submit + stream synchronisation, event = hipEvent time around the submission, in the stream (what the
// benchmark reports); medians of N repetitions after a warm-up of 1.5 s. The graph building (capture, instantiate) is not timed.
#include <hip/hip_runtime.h>

#include <fused_kernel/algorithms/algorithms.h>
#include <fused_kernel/core/data/ptr_nd.h>
#include <fused_kernel/core/data/ptr_utils.h>
#include <fused_kernel/fused_kernel.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <string>
#include <vector>

#define HC(x) do { const hipError_t e_ = (x); if (e_ != hipSuccess) { std::fprintf(stderr, "%s failed at line %d: %s\n", #x, __LINE__, hipGetErrorString(e_)); std::exit(1); } } while (0)

using Clock = std::chrono::steady_clock;
static double msSince(const Clock::time_point a, const Clock::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); }
static double med(std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }

struct Result { double submit, total, event; };

static Result timeIt(const std::function<void()>& submit, hipStream_t hs, const int reps, const double warmupSeconds) {
    hipEvent_t e0, e1;
    HC(hipEventCreate(&e0));
    HC(hipEventCreate(&e1));
    const auto t0 = Clock::now();
    while (msSince(t0, Clock::now()) < warmupSeconds * 1000.0) { submit(); HC(hipStreamSynchronize(hs)); }
    std::vector<double> sub, tot, ev;
    for (int i = 0; i < reps; ++i) {
        const auto a = Clock::now();
        HC(hipEventRecord(e0, hs));
        submit();
        HC(hipEventRecord(e1, hs));
        const auto b = Clock::now();
        HC(hipStreamSynchronize(hs));
        const auto c = Clock::now();
        float f = 0;
        HC(hipEventElapsedTime(&f, e0, e1));
        sub.push_back(msSince(a, b));
        tot.push_back(msSince(a, c));
        ev.push_back(f);
    }
    HC(hipEventDestroy(e0));
    HC(hipEventDestroy(e1));
    return {med(sub), med(tot), med(ev)};
}

static bool wanted(const char* name) {
    const char* v = std::getenv("PROBE_VARIANT");
    return v == nullptr || std::string(v) == name;
}
static double warmupSeconds() {
    const char* v = std::getenv("PROBE_WARMUP");
    return v ? std::atof(v) : 1.5;
}

template <size_t BATCH>
void probe(fk::Stream& stream, const int reps) {
    using InputType = uchar;
    using OutputType = float;
    const fk::Size cropSize(60, 120);
    fk::Ptr2D<InputType> d_input(3840, 2160);
    fk::setTo(static_cast<InputType>(10), d_input, stream);
    std::array<fk::Ptr2D<OutputType>, BATCH> d_output;
    std::array<fk::Ptr2D<InputType>, BATCH> crops;
    for (int i = 0; i < static_cast<int>(BATCH); i++) {
        crops[i] = d_input.crop(fk::Point{i, i, 0}, fk::PtrDims<fk::ND::_2D>{static_cast<uint>(cropSize.width), static_cast<uint>(cropSize.height), static_cast<uint>(d_input.dims().pitch)});
        d_output[i].Alloc(cropSize);
    }
    const auto read_array = fk::PerThreadRead<fk::ND::_2D, InputType>::build_batch(crops);
    const auto saturate = fk::SaturateCast<InputType, OutputType>::build();
    const auto mul = fk::Mul<OutputType>::build(1.0f);
    const auto sub = fk::Sub<OutputType>::build(1.f);
    const auto div = fk::Div<OutputType>::build(3.2f);
    const auto write_array = fk::PerThreadWrite<fk::ND::_2D, OutputType>::build_batch(d_output);
    stream.sync();
    const hipStream_t hs = stream.getHIPStream();

    const auto launchOne = [&](fk::Stream& s, const int i) { fk::executeOperations<fk::TransformDPP<>>(s, read_array[i], saturate, mul, sub, div, write_array[i]); };

    // plain launches
    Result plain{0, 0, 0}, children{0, 0, 0}, childrenSeq{0, 0, 0}, flat{0, 0, 0};
    if (wanted("plain")) plain = timeIt([&] { for (int i = 0; i < static_cast<int>(BATCH); ++i) launchOne(stream, i); }, hs, reps, warmupSeconds());

    // children (the graph of the benchmark) and children_seq
    const auto buildChildren = [&](const bool sequential) {
        hipGraph_t mainGraph;
        HC(hipGraphCreate(&mainGraph, 0));
        std::vector<hipGraphNode_t> nodes(BATCH);
        for (int i = 0; i < static_cast<int>(BATCH); i++) {
            fk::Stream nodeStream;
            hipGraph_t subGraph;
            HC(hipStreamBeginCapture(nodeStream, hipStreamCaptureModeGlobal));
            launchOne(nodeStream, i);
            HC(hipStreamEndCapture(nodeStream, &subGraph));
            if (sequential && i > 0) {
                HC(hipGraphAddChildGraphNode(&nodes[i], mainGraph, &nodes[i - 1], 1, subGraph));
            } else {
                HC(hipGraphAddChildGraphNode(&nodes[i], mainGraph, nullptr, 0, subGraph));
            }
            HC(hipGraphDestroy(subGraph));
        }
        hipGraphExec_t exec;
        HC(hipGraphInstantiate(&exec, mainGraph, NULL, NULL, 0));
        HC(hipGraphDestroy(mainGraph));
        return exec;
    };
    if (wanted("children")) {
        hipGraphExec_t childrenExec = buildChildren(false);
        children = timeIt([&] { HC(hipGraphLaunch(childrenExec, hs)); }, hs, reps, warmupSeconds());
        HC(hipGraphExecDestroy(childrenExec));
    }
    if (wanted("children_seq")) {
        hipGraphExec_t seqExec = buildChildren(true);
        childrenSeq = timeIt([&] { HC(hipGraphLaunch(seqExec, hs)); }, hs, reps, warmupSeconds());
        HC(hipGraphExecDestroy(seqExec));
    }

    // flat: one capture of all the launches
    if (wanted("flat")) {
    hipGraphExec_t flatExec;
    {
        fk::Stream captureStream;
        hipGraph_t g;
        HC(hipStreamBeginCapture(captureStream, hipStreamCaptureModeGlobal));
        for (int i = 0; i < static_cast<int>(BATCH); ++i) launchOne(captureStream, i);
        HC(hipStreamEndCapture(captureStream, &g));
        HC(hipGraphInstantiate(&flatExec, g, NULL, NULL, 0));
        HC(hipGraphDestroy(g));
    }
    flat = timeIt([&] { HC(hipGraphLaunch(flatExec, hs)); }, hs, reps, warmupSeconds());
    HC(hipGraphExecDestroy(flatExec));
    }

    std::printf("batch %4zu | plain: submit %.4f total %.4f event %.4f | children: submit %.4f total %.4f event %.4f | children_seq: submit %.4f total %.4f event %.4f | flat: submit %.4f total %.4f event %.4f   (ms)\n",
                BATCH, plain.submit, plain.total, plain.event, children.submit, children.total, children.event, childrenSeq.submit, childrenSeq.total, childrenSeq.event, flat.submit, flat.total, flat.event);
    std::fflush(stdout);
}

int main(int argc, char** argv) {
    const int reps = argc > 1 ? std::atoi(argv[1]) : 100;
    fk::Stream stream;
    if (argc > 2) {
        const int b = std::atoi(argv[2]);
        if (b == 101) probe<101>(stream, reps);
        else if (b == 301) probe<301>(stream, reps);
        else if (b == 1191) probe<1191>(stream, reps);
        return 0;
    }
    probe<11>(stream, reps);
    probe<51>(stream, reps);
    probe<101>(stream, reps);
    probe<201>(stream, reps);
    probe<301>(stream, reps);
    probe<601>(stream, reps);
    probe<1191>(stream, reps);
    return 0;
}
