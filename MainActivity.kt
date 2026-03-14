package com.broot

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.media.MediaCodec
import android.media.MediaExtractor
import android.media.MediaFormat
import android.net.Uri
import android.os.Build
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.ContextCompat
import kotlinx.coroutines.*
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.*

// ── Data ─────────────────────────────────────────────────────────────────────

data class BarResult(val barIndex: Int, val rootNote: String, val confidence: Float)
data class AnalysisResult(val bpm: Float, val bars: List<BarResult>)

sealed class AnalysisState {
    object Idle : AnalysisState()
    data class Analyzing(val progress: Float, val status: String) : AnalysisState()
    data class Done(val result: AnalysisResult, val fileName: String) : AnalysisState()
    data class Error(val message: String) : AnalysisState()
}

// ── Note utilities ────────────────────────────────────────────────────────────

private val NOTE_NAMES = arrayOf("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")

private fun noteColor(note: String): Color = when (note) {
    "C"  -> Color(0xFFEF5350)
    "C#" -> Color(0xFFEC407A)
    "D"  -> Color(0xFFAB47BC)
    "D#" -> Color(0xFF7E57C2)
    "E"  -> Color(0xFF42A5F5)
    "F"  -> Color(0xFF26C6DA)
    "F#" -> Color(0xFF26A69A)
    "G"  -> Color(0xFF66BB6A)
    "G#" -> Color(0xFFD4E157)
    "A"  -> Color(0xFFFFA726)
    "A#" -> Color(0xFFFF7043)
    "B"  -> Color(0xFF8D6E63)
    else -> Color(0xFF616161)
}

private fun freqToNote(freq: Double): String {
    if (freq <= 0.0 || freq.isNaN() || freq.isInfinite()) return "?"
    val noteNum = 12.0 * log2(freq / 440.0) + 69.0
    val idx = ((noteNum.roundToInt() % 12) + 12) % 12
    return NOTE_NAMES[idx]
}

// ── FFT ───────────────────────────────────────────────────────────────────────

private fun fft(re: DoubleArray, im: DoubleArray) {
    val n = re.size
    var j = 0
    for (i in 1 until n) {
        var bit = n ushr 1
        while (j and bit != 0) { j = j xor bit; bit = bit ushr 1 }
        j = j xor bit
        if (i < j) {
            var t = re[i]; re[i] = re[j]; re[j] = t
            t = im[i]; im[i] = im[j]; im[j] = t
        }
    }
    var len = 2
    while (len <= n) {
        val half = len / 2
        val ang = -2.0 * PI / len
        val wRe = cos(ang); val wIm = sin(ang)
        var i = 0
        while (i < n) {
            var cRe = 1.0; var cIm = 0.0
            for (jj in 0 until half) {
                val uRe = re[i+jj]; val uIm = im[i+jj]
                val vRe = re[i+jj+half]*cRe - im[i+jj+half]*cIm
                val vIm = re[i+jj+half]*cIm + im[i+jj+half]*cRe
                re[i+jj] = uRe+vRe; im[i+jj] = uIm+vIm
                re[i+jj+half] = uRe-vRe; im[i+jj+half] = uIm-vIm
                val nRe = cRe*wRe - cIm*wIm; cIm = cRe*wIm + cIm*wRe; cRe = nRe
            }
            i += len
        }
        len *= 2
    }
}

private fun magnitudeSpectrum(samples: FloatArray, offset: Int, fftSize: Int): DoubleArray {
    val re = DoubleArray(fftSize); val im = DoubleArray(fftSize)
    val count = minOf(samples.size - offset, fftSize)
    for (i in 0 until count) {
        val w = 0.5 * (1.0 - cos(2.0 * PI * i / (fftSize - 1)))
        re[i] = samples[offset + i].toDouble() * w
    }
    fft(re, im)
    return DoubleArray(fftSize / 2) { i -> sqrt(re[i]*re[i] + im[i]*im[i]) }
}

// ── Audio decoding ────────────────────────────────────────────────────────────

private suspend fun decodeAudio(context: Context, uri: Uri): Pair<Int, FloatArray> =
    withContext(Dispatchers.IO) {
        val extractor = MediaExtractor()
        extractor.setDataSource(context, uri, null)

        var trackIdx = -1
        var trackFormat: MediaFormat? = null
        for (i in 0 until extractor.trackCount) {
            val fmt = extractor.getTrackFormat(i)
            if (fmt.getString(MediaFormat.KEY_MIME)?.startsWith("audio/") == true) {
                trackIdx = i; trackFormat = fmt; break
            }
        }
        if (trackIdx < 0 || trackFormat == null) error("No audio track found in file")

        extractor.selectTrack(trackIdx)
        val mime       = trackFormat.getString(MediaFormat.KEY_MIME)!!
        val sampleRate = trackFormat.getInteger(MediaFormat.KEY_SAMPLE_RATE)
        val channels   = trackFormat.getInteger(MediaFormat.KEY_CHANNEL_COUNT)

        val codec = MediaCodec.createDecoderByType(mime)
        codec.configure(trackFormat, null, null, 0)
        codec.start()

        val chunks = mutableListOf<ByteArray>()
        val info = MediaCodec.BufferInfo()
        var inputDone = false; var outputDone = false
        val timeout = 10_000L

        while (!outputDone) {
            if (!inputDone) {
                val idx = codec.dequeueInputBuffer(timeout)
                if (idx >= 0) {
                    val buf = codec.getInputBuffer(idx)!!
                    val n = extractor.readSampleData(buf, 0)
                    if (n < 0) {
                        codec.queueInputBuffer(idx, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                        inputDone = true
                    } else {
                        codec.queueInputBuffer(idx, 0, n, extractor.sampleTime, 0)
                        extractor.advance()
                    }
                }
            }
            when (val out = codec.dequeueOutputBuffer(info, timeout)) {
                MediaCodec.INFO_TRY_AGAIN_LATER -> { /* wait */ }
                MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> { /* ignore */ }
                else -> if (out >= 0) {
                    if (info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) outputDone = true
                    val buf = codec.getOutputBuffer(out)!!
                    buf.position(info.offset); buf.limit(info.offset + info.size)
                    val bytes = ByteArray(info.size); buf.get(bytes)
                    chunks.add(bytes)
                    codec.releaseOutputBuffer(out, false)
                }
            }
        }
        codec.stop(); codec.release(); extractor.release()

        val totalBytes = chunks.sumOf { it.size }
        val combined = ByteArray(totalBytes)
        var pos = 0
        for (c in chunks) { c.copyInto(combined, pos); pos += c.size }

        val shortBuf = ByteBuffer.wrap(combined).order(ByteOrder.LITTLE_ENDIAN).asShortBuffer()
        val totalShorts = totalBytes / 2
        val mono = FloatArray(totalShorts / channels)
        for (i in mono.indices) {
            var sum = 0.0f
            for (ch in 0 until channels) sum += shortBuf.get()
            mono[i] = sum / channels / 32768.0f
        }
        Pair(sampleRate, mono)
    }

// ── BPM detection ─────────────────────────────────────────────────────────────

private fun detectBPM(samples: FloatArray, sampleRate: Int): Float {
    val frameSize = 1024; val hopSize = 512
    val frameCount = (samples.size - frameSize) / hopSize
    if (frameCount < 4) return 120f

    val energy = FloatArray(frameCount)
    for (f in 0 until frameCount) {
        var e = 0.0f
        for (s in 0 until frameSize) { val x = samples[f*hopSize+s]; e += x*x }
        energy[f] = e / frameSize
    }
    val onset = FloatArray(frameCount)
    for (f in 1 until frameCount) onset[f] = maxOf(0f, energy[f] - energy[f-1])

    val maxLag = (sampleRate * 60.0 / (hopSize * 60.0)).toInt().coerceAtLeast(1)
    val minLag = (sampleRate * 60.0 / (hopSize * 200.0)).toInt().coerceAtLeast(1)

    var bestLag = minLag; var bestCorr = -1.0
    for (lag in minLag..maxLag) {
        var corr = 0.0
        for (f in 0 until frameCount - lag) corr += onset[f] * onset[f+lag]
        if (corr > bestCorr) { bestCorr = corr; bestLag = lag }
    }
    return (sampleRate * 60.0 / (hopSize * bestLag)).toFloat().coerceIn(60f, 200f)
}

// ── Root note per bar ─────────────────────────────────────────────────────────

private const val FFT_SIZE = 8192

private fun analyzeRootNote(samples: FloatArray, start: Int, end: Int, sampleRate: Int): Pair<String, Float> {
    if (end - start < FFT_SIZE) return Pair("?", 0f)

    val mag = DoubleArray(FFT_SIZE / 2)
    var count = 0
    var p = start
    while (p + FFT_SIZE <= end) {
        val m = magnitudeSpectrum(samples, p, FFT_SIZE)
        for (i in m.indices) mag[i] += m[i]
        p += FFT_SIZE / 2; count++
    }
    if (count == 0) return Pair("?", 0f)
    for (i in mag.indices) mag[i] /= count.toDouble()

    val loIdx = (60.0  * FFT_SIZE / sampleRate).toInt().coerceAtLeast(1)
    val hiIdx = (500.0 * FFT_SIZE / sampleRate).toInt().coerceAtMost(FFT_SIZE/2 - 1)

    var peakIdx = loIdx; var peakMag = 0.0
    for (i in loIdx..hiIdx) if (mag[i] > peakMag) { peakMag = mag[i]; peakIdx = i }

    val freq = peakIdx.toDouble() * sampleRate / FFT_SIZE
    val avgMag = (loIdx..hiIdx).sumOf { mag[it] } / (hiIdx - loIdx + 1)
    val conf = if (avgMag > 0) (peakMag / avgMag / 20.0).toFloat().coerceIn(0f, 1f) else 0f

    return Pair(freqToNote(freq), conf)
}

// ── Analysis pipeline ─────────────────────────────────────────────────────────

private suspend fun analyzeAudio(
    context: Context,
    uri: Uri,
    onProgress: (Float, String) -> Unit
): AnalysisResult = withContext(Dispatchers.Default) {
    onProgress(0.05f, "Decoding audio…")
    val (sampleRate, samples) = withContext(Dispatchers.IO) { decodeAudio(context, uri) }

    onProgress(0.35f, "Detecting BPM…")
    val bpm = detectBPM(samples, sampleRate)

    val barSamples = ((60.0 / bpm) * 4 * sampleRate).toInt()
    val totalBars  = (samples.size / barSamples).coerceAtLeast(1)
    onProgress(0.45f, "BPM ≈ ${bpm.toInt()} · $totalBars bars detected")

    val bars = mutableListOf<BarResult>()
    for (b in 0 until totalBars) {
        val s = b * barSamples
        val e = minOf(s + barSamples, samples.size)
        val (note, conf) = analyzeRootNote(samples, s, e, sampleRate)
        bars.add(BarResult(b, note, conf))
        onProgress(0.45f + 0.50f * (b + 1f) / totalBars, "Bar ${b+1} / $totalBars")
        yield()
    }
    AnalysisResult(bpm, bars)
}

// ── Activity ──────────────────────────────────────────────────────────────────

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            MaterialTheme(
                colorScheme = darkColorScheme(
                    primary    = Color(0xFF6650A4),
                    background = Color(0xFF0D0D0D),
                    surface    = Color(0xFF1A1A1A)
                )
            ) {
                Surface(Modifier.fillMaxSize()) { BrootApp() }
            }
        }
    }
}

// ── Composables ───────────────────────────────────────────────────────────────

@Composable
fun BrootApp() {
    val ctx   = LocalContext.current
    val scope = rememberCoroutineScope()
    var state by remember { mutableStateOf<AnalysisState>(AnalysisState.Idle) }

    val permLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { }

    val filePicker = rememberLauncherForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri ->
        if (uri == null) return@rememberLauncherForActivityResult
        val name = uri.lastPathSegment ?: "audio"
        scope.launch {
            state = AnalysisState.Analyzing(0f, "Starting…")
            runCatching {
                analyzeAudio(ctx, uri) { p, m -> state = AnalysisState.Analyzing(p, m) }
            }.onSuccess { state = AnalysisState.Done(it, name) }
             .onFailure { state = AnalysisState.Error(it.message ?: "Unknown error") }
        }
    }

    fun pickFile() {
        val perms = if (Build.VERSION.SDK_INT >= 33)
            arrayOf(Manifest.permission.READ_MEDIA_AUDIO)
        else arrayOf(Manifest.permission.READ_EXTERNAL_STORAGE)
        val allGranted = perms.all {
            ContextCompat.checkSelfPermission(ctx, it) == PackageManager.PERMISSION_GRANTED
        }
        if (!allGranted) permLauncher.launch(perms)
        scope.launch { delay(300); filePicker.launch("audio/*") }
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(Color(0xFF0D0D0D))
            .padding(top = 52.dp, start = 16.dp, end = 16.dp, bottom = 24.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Text("broot", fontSize = 28.sp, fontWeight = FontWeight.Bold, color = Color.White,
            letterSpacing = 4.sp)
        Text("bar · root · analyzer", fontSize = 11.sp, color = Color(0xFF444444), letterSpacing = 2.sp)
        Spacer(Modifier.height(40.dp))
        when (val s = state) {
            is AnalysisState.Idle      -> PickButton(::pickFile)
            is AnalysisState.Analyzing -> AnalyzingView(s.progress, s.status)
            is AnalysisState.Done      -> DoneView(s.result, s.fileName) { state = AnalysisState.Idle }
            is AnalysisState.Error     -> ErrorView(s.message) { state = AnalysisState.Idle }
        }
    }
}

@Composable
fun PickButton(onClick: () -> Unit) {
    Button(
        onClick = onClick,
        modifier = Modifier.fillMaxWidth(0.72f).height(54.dp),
        colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF6650A4)),
        shape  = RoundedCornerShape(14.dp)
    ) { Text("Pick Audio File", fontSize = 16.sp, fontWeight = FontWeight.SemiBold) }
}

@Composable
fun AnalyzingView(progress: Float, status: String) {
    Column(horizontalAlignment = Alignment.CenterHorizontally) {
        LinearProgressIndicator(
            progress    = { progress },
            modifier    = Modifier.fillMaxWidth().height(6.dp).clip(RoundedCornerShape(3.dp)),
            color       = Color(0xFF6650A4),
            trackColor  = Color(0xFF2A2A2A)
        )
        Spacer(Modifier.height(14.dp))
        Text(status, fontSize = 14.sp, color = Color(0xFF999999))
        Spacer(Modifier.height(4.dp))
        Text("${(progress * 100).toInt()}%", fontSize = 11.sp, color = Color(0xFF555555))
    }
}

@Composable
fun DoneView(result: AnalysisResult, fileName: String, onReset: () -> Unit) {
    Column(horizontalAlignment = Alignment.CenterHorizontally) {
        Row(horizontalArrangement = Arrangement.spacedBy(40.dp)) {
            InfoPill("BPM", "${result.bpm.toInt()}")
            InfoPill("BARS", "${result.bars.size}")
        }
        Spacer(Modifier.height(6.dp))
        Text(fileName.substringAfterLast("/"), fontSize = 11.sp, color = Color(0xFF333333), maxLines = 1)
        Spacer(Modifier.height(28.dp))

        Text("T I M E L I N E", fontSize = 10.sp, color = Color(0xFF444444), letterSpacing = 3.sp,
            modifier = Modifier.align(Alignment.Start).padding(start = 4.dp, bottom = 10.dp))
        LazyRow(
            horizontalArrangement = Arrangement.spacedBy(6.dp),
            contentPadding = PaddingValues(horizontal = 4.dp)
        ) {
            items(result.bars) { bar -> BarCard(bar) }
        }

        Spacer(Modifier.height(22.dp))
        val uniqueNotes = result.bars.map { it.rootNote }.distinct().filter { it != "?" }.sorted()
        if (uniqueNotes.isNotEmpty()) {
            Text("N O T E S", fontSize = 10.sp, color = Color(0xFF444444), letterSpacing = 3.sp,
                modifier = Modifier.align(Alignment.Start).padding(start = 4.dp, bottom = 8.dp))
            Row(
                horizontalArrangement = Arrangement.spacedBy(6.dp),
                modifier = Modifier.horizontalScroll(rememberScrollState()).align(Alignment.Start)
            ) {
                uniqueNotes.forEach { note ->
                    Box(
                        contentAlignment = Alignment.Center,
                        modifier = Modifier.size(36.dp).clip(RoundedCornerShape(8.dp))
                            .background(noteColor(note))
                    ) { Text(note, fontSize = 10.sp, fontWeight = FontWeight.Bold, color = Color.White) }
                }
            }
        }

        Spacer(Modifier.height(30.dp))
        TextButton(onClick = onReset) {
            Text("Analyze Another", color = Color(0xFF6650A4), letterSpacing = 1.sp)
        }
    }
}

@Composable
fun BarCard(bar: BarResult) {
    val color = noteColor(bar.rootNote)
    Column(horizontalAlignment = Alignment.CenterHorizontally, modifier = Modifier.width(54.dp)) {
        Text("${bar.barIndex + 1}", fontSize = 10.sp, color = Color(0xFF444444), textAlign = TextAlign.Center)
        Spacer(Modifier.height(4.dp))
        Box(
            contentAlignment = Alignment.Center,
            modifier = Modifier
                .fillMaxWidth()
                .height(68.dp)
                .clip(RoundedCornerShape(10.dp))
                .background(color.copy(alpha = 0.88f))
        ) { Text(bar.rootNote, fontSize = 17.sp, fontWeight = FontWeight.Bold, color = Color.White) }
        Spacer(Modifier.height(4.dp))
        Box(modifier = Modifier.size(4.dp).clip(RoundedCornerShape(2.dp))
            .background(color.copy(alpha = bar.confidence.coerceIn(0.15f, 1f))))
    }
}

@Composable
fun InfoPill(label: String, value: String) {
    Column(horizontalAlignment = Alignment.CenterHorizontally) {
        Text(value, fontSize = 30.sp, fontWeight = FontWeight.Bold, color = Color.White)
        Text(label, fontSize = 10.sp, color = Color(0xFF444444), letterSpacing = 2.sp)
    }
}

@Composable
fun ErrorView(message: String, onReset: () -> Unit) {
    Column(horizontalAlignment = Alignment.CenterHorizontally) {
        Text("Analysis failed", fontSize = 16.sp, color = Color(0xFFEF5350), fontWeight = FontWeight.SemiBold)
        Spacer(Modifier.height(10.dp))
        Text(message, fontSize = 13.sp, color = Color(0xFF777777), textAlign = TextAlign.Center)
        Spacer(Modifier.height(22.dp))
        TextButton(onClick = onReset) { Text("Try Again", color = Color(0xFF6650A4)) }
    }
}
