package com.broot

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.media.MediaCodec
import android.media.MediaExtractor
import android.media.MediaFormat
import android.media.MediaPlayer
import android.media.audiofx.Equalizer
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Environment
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyListState
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.itemsIndexed
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
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
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.*

// ── Data ─────────────────────────────────────────────────────────────────────

data class BarResult(val barIndex: Int, val rootNote: String, val confidence: Float)
data class AnalysisResult(val bpm: Float, val bars: List<BarResult>, val sampleRate: Int)

sealed class AnalysisState {
    object Idle : AnalysisState()
    data class Analyzing(val progress: Float, val status: String) : AnalysisState()
    data class Done(val result: AnalysisResult, val fileName: String, val uri: Uri) : AnalysisState()
    data class Error(val message: String) : AnalysisState()
}

data class EqBandInfo(val band: Int, val centerFreqHz: Int, val minLevel: Int, val maxLevel: Int)

// ── Note / color utilities ────────────────────────────────────────────────────

private val NOTE_NAMES = arrayOf("C","C#","D","D#","E","F","F#","G","G#","A","A#","B")

// Base frequencies at octave 2 (~65 Hz range)
private val BASS_FREQ_BASE = mapOf(
    "C"  to 65.41,  "C#" to 69.30,  "D"  to 73.42,  "D#" to 77.78,
    "E"  to 82.41,  "F"  to 87.31,  "F#" to 92.50,  "G"  to 98.00,
    "G#" to 103.83, "A"  to 110.00, "A#" to 116.54,  "B"  to 123.47
)

// Octave multipliers: octave 1 = sub-bass, 2 = deep bass, 3 = mid-bass, 4 = upper-bass
private val OCTAVE_MULT = mapOf(1 to 0.5, 2 to 1.0, 3 to 2.0, 4 to 4.0)
private val OCTAVE_LABEL = mapOf(1 to "SUB", 2 to "DEEP", 3 to "MID", 4 to "HIGH")

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

private fun formatMs(ms: Int): String {
    val s = ms / 1000
    return "%d:%02d".format(s / 60, s % 60)
}

private fun formatFreq(hz: Int): String = if (hz >= 1000) "${hz / 1000}k" else "$hz"

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
                val uRe = re[i + jj]; val uIm = im[i + jj]
                val vRe = re[i + jj + half] * cRe - im[i + jj + half] * cIm
                val vIm = re[i + jj + half] * cIm + im[i + jj + half] * cRe
                re[i + jj] = uRe + vRe; im[i + jj] = uIm + vIm
                re[i + jj + half] = uRe - vRe; im[i + jj + half] = uIm - vIm
                val nRe = cRe * wRe - cIm * wIm; cIm = cRe * wIm + cIm * wRe; cRe = nRe
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
    return DoubleArray(fftSize / 2) { i -> sqrt(re[i] * re[i] + im[i] * im[i]) }
}

// ── Audio decoding ────────────────────────────────────────────────────────────

private suspend fun decodeAudio(context: Context, uri: Uri): Pair<Int, FloatArray> =
    withContext(Dispatchers.IO) {
        val extractor = MediaExtractor()
        extractor.setDataSource(context, uri, null)

        var trackIdx = -1; var trackFormat: MediaFormat? = null
        for (i in 0 until extractor.trackCount) {
            val fmt = extractor.getTrackFormat(i)
            if (fmt.getString(MediaFormat.KEY_MIME)?.startsWith("audio/") == true) {
                trackIdx = i; trackFormat = fmt; break
            }
        }
        if (trackIdx < 0 || trackFormat == null) error("No audio track found")

        extractor.selectTrack(trackIdx)
        val mime       = trackFormat.getString(MediaFormat.KEY_MIME)!!
        val sampleRate = trackFormat.getInteger(MediaFormat.KEY_SAMPLE_RATE)
        val channels   = trackFormat.getInteger(MediaFormat.KEY_CHANNEL_COUNT)

        val codec = MediaCodec.createDecoderByType(mime)
        codec.configure(trackFormat, null, null, 0)
        codec.start()

        val chunks = mutableListOf<ByteArray>()
        val info   = MediaCodec.BufferInfo()
        var inputDone = false; var outputDone = false

        while (!outputDone) {
            if (!inputDone) {
                val idx = codec.dequeueInputBuffer(10_000L)
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
            when (val out = codec.dequeueOutputBuffer(info, 10_000L)) {
                MediaCodec.INFO_TRY_AGAIN_LATER      -> {}
                MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> {}
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
        val combined   = ByteArray(totalBytes)
        var pos = 0
        for (c in chunks) { c.copyInto(combined, pos); pos += c.size }

        val shortBuf = ByteBuffer.wrap(combined).order(ByteOrder.LITTLE_ENDIAN).asShortBuffer()
        val mono     = FloatArray(totalBytes / 2 / channels)
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
        for (s in 0 until frameSize) { val x = samples[f * hopSize + s]; e += x * x }
        energy[f] = e / frameSize
    }
    val onset = FloatArray(frameCount)
    for (f in 1 until frameCount) onset[f] = maxOf(0f, energy[f] - energy[f - 1])

    val maxLag = (sampleRate * 60.0 / (hopSize * 60.0)).toInt().coerceAtLeast(1)
    val minLag = (sampleRate * 60.0 / (hopSize * 200.0)).toInt().coerceAtLeast(1)

    var bestLag = minLag; var bestCorr = -1.0
    for (lag in minLag..maxLag) {
        var corr = 0.0
        for (f in 0 until frameCount - lag) corr += (onset[f] * onset[f + lag]).toDouble()
        if (corr > bestCorr) { bestCorr = corr; bestLag = lag }
    }
    return (sampleRate * 60.0 / (hopSize * bestLag)).toFloat().coerceIn(60f, 200f)
}

// ── Root note per bar ─────────────────────────────────────────────────────────

private const val FFT_SIZE = 8192

private fun analyzeRootNote(samples: FloatArray, start: Int, end: Int, sampleRate: Int): Pair<String, Float> {
    if (end - start < FFT_SIZE) return Pair("?", 0f)

    val mag = DoubleArray(FFT_SIZE / 2)
    var count = 0; var p = start
    while (p + FFT_SIZE <= end) {
        val m = magnitudeSpectrum(samples, p, FFT_SIZE)
        for (i in m.indices) mag[i] += m[i]
        p += FFT_SIZE / 2; count++
    }
    if (count == 0) return Pair("?", 0f)
    for (i in mag.indices) mag[i] /= count.toDouble()

    val loIdx = (60.0  * FFT_SIZE / sampleRate).toInt().coerceAtLeast(1)
    val hiIdx = (500.0 * FFT_SIZE / sampleRate).toInt().coerceAtMost(FFT_SIZE / 2 - 1)

    var peakIdx = loIdx; var peakMag = 0.0
    for (i in loIdx..hiIdx) if (mag[i] > peakMag) { peakMag = mag[i]; peakIdx = i }

    val freq   = peakIdx.toDouble() * sampleRate / FFT_SIZE
    val avgMag = (loIdx..hiIdx).sumOf { mag[it] } / (hiIdx - loIdx + 1)
    val conf   = if (avgMag > 0) (peakMag / avgMag / 20.0).toFloat().coerceIn(0f, 1f) else 0f

    return Pair(freqToNote(freq), conf)
}

// ── Bass synthesis ────────────────────────────────────────────────────────────

// Pure sine wave per bar at chosen octave with smooth attack/release envelope
private fun generateBassLine(
    bars: List<BarResult>,
    enabled: List<Boolean>,
    barSamples: Int,
    sampleRate: Int,
    octave: Int       // 1=sub, 2=deep, 3=mid, 4=high
): FloatArray {
    val mult = OCTAVE_MULT[octave] ?: 1.0
    val out  = FloatArray(bars.size * barSamples)
    for ((b, bar) in bars.withIndex()) {
        if (!enabled[b] || bar.rootNote == "?") continue
        val baseFreq = BASS_FREQ_BASE[bar.rootNote] ?: continue
        val freq     = baseFreq * mult
        val start    = b * barSamples
        val attackSamples  = (sampleRate * 0.015).toInt().coerceAtLeast(1) // 15 ms attack
        val releaseSamples = (sampleRate * 0.040).toInt().coerceAtLeast(1) // 40 ms release
        for (i in 0 until barSamples) {
            val t       = i.toDouble() / sampleRate
            val attack  = minOf(1.0, i.toDouble() / attackSamples)
            val release = minOf(1.0, (barSamples - i).toDouble() / releaseSamples)
            val env     = attack * release
            out[start + i] = (sin(2.0 * PI * freq * t) * env * 0.65).toFloat()
        }
    }
    return out
}

// ── WAV export ────────────────────────────────────────────────────────────────

private fun writeWav(file: File, samples: FloatArray, sampleRate: Int) {
    FileOutputStream(file).use { fos ->
        val dataSize = samples.size * 2
        val header = ByteBuffer.allocate(44).order(ByteOrder.LITTLE_ENDIAN).apply {
            put("RIFF".toByteArray()); putInt(36 + dataSize)
            put("WAVE".toByteArray()); put("fmt ".toByteArray())
            putInt(16); putShort(1); putShort(1)
            putInt(sampleRate); putInt(sampleRate * 2); putShort(2); putShort(16)
            put("data".toByteArray()); putInt(dataSize)
        }
        fos.write(header.array())
        val buf = ByteBuffer.allocate(4096).order(ByteOrder.LITTLE_ENDIAN)
        var i = 0
        while (i < samples.size) {
            buf.clear()
            while (buf.remaining() >= 2 && i < samples.size)
                buf.putShort((samples[i++].coerceIn(-1f, 1f) * 32767).toInt().toShort())
            buf.flip()
            val arr = ByteArray(buf.limit()); buf.get(arr); fos.write(arr)
        }
    }
}

private suspend fun saveMix(
    context: Context, uri: Uri,
    result: AnalysisResult, enabled: List<Boolean>, octave: Int,
    onStatus: (String) -> Unit
): String = withContext(Dispatchers.IO) {
    onStatus("Decoding audio…")
    val (sr, original) = decodeAudio(context, uri)
    onStatus("Generating bass…")
    val barSamples = ((60.0 / result.bpm) * 4 * sr).toInt()
    val bass = generateBassLine(result.bars, enabled, barSamples, sr, octave)
    onStatus("Mixing…")
    val mixed = FloatArray(maxOf(original.size, bass.size)) { i ->
        val a = if (i < original.size) original[i] else 0f
        val b = if (i < bass.size) bass[i] else 0f
        (a + b).coerceIn(-1f, 1f)
    }
    onStatus("Writing WAV…")
    val dir  = context.getExternalFilesDir(Environment.DIRECTORY_MUSIC) ?: context.filesDir
    dir.mkdirs()
    val file = File(dir, "broot_${System.currentTimeMillis()}.wav")
    writeWav(file, mixed, sr)
    file.absolutePath
}

// ── Analysis pipeline ─────────────────────────────────────────────────────────

private suspend fun analyzeAudio(
    context: Context, uri: Uri,
    onProgress: (Float, String) -> Unit
): AnalysisResult = withContext(Dispatchers.Default) {
    onProgress(0.05f, "Decoding audio…")
    val (sampleRate, samples) = withContext(Dispatchers.IO) { decodeAudio(context, uri) }

    onProgress(0.35f, "Detecting BPM…")
    val bpm = detectBPM(samples, sampleRate)

    val barSamples = ((60.0 / bpm) * 4 * sampleRate).toInt()
    val totalBars  = (samples.size / barSamples).coerceAtLeast(1)
    onProgress(0.45f, "BPM ≈ ${bpm.toInt()} · $totalBars bars")

    val bars = mutableListOf<BarResult>()
    for (b in 0 until totalBars) {
        val s = b * barSamples
        val e = minOf(s + barSamples, samples.size)
        val (note, conf) = analyzeRootNote(samples, s, e, sampleRate)
        bars.add(BarResult(b, note, conf))
        onProgress(0.45f + 0.50f * (b + 1f) / totalBars, "Bar ${b + 1} / $totalBars")
        yield()
    }
    AnalysisResult(bpm, bars, sampleRate)
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

// ── BrootApp ──────────────────────────────────────────────────────────────────

@Composable
fun BrootApp() {
    val ctx   = LocalContext.current
    val scope = rememberCoroutineScope()
    var state by remember { mutableStateOf<AnalysisState>(AnalysisState.Idle) }

    val permLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()) { }

    val filePicker = rememberLauncherForActivityResult(
        ActivityResultContracts.GetContent()) { uri ->
        if (uri == null) return@rememberLauncherForActivityResult
        val name = uri.lastPathSegment ?: "audio"
        scope.launch {
            state = AnalysisState.Analyzing(0f, "Starting…")
            runCatching {
                analyzeAudio(ctx, uri) { p, m -> state = AnalysisState.Analyzing(p, m) }
            }.onSuccess { state = AnalysisState.Done(it, name, uri) }
             .onFailure { state = AnalysisState.Error(it.message ?: "Unknown error") }
        }
    }

    fun pickFile() {
        val perms = if (Build.VERSION.SDK_INT >= 33)
            arrayOf(Manifest.permission.READ_MEDIA_AUDIO)
        else arrayOf(Manifest.permission.READ_EXTERNAL_STORAGE)
        if (perms.any { ContextCompat.checkSelfPermission(ctx, it) != PackageManager.PERMISSION_GRANTED })
            permLauncher.launch(perms)
        scope.launch { delay(300); filePicker.launch("audio/*") }
    }

    val s = state
    if (s is AnalysisState.Done) {
        DoneView(s.result, s.uri, s.fileName) { state = AnalysisState.Idle }
        return
    }

    Column(
        modifier = Modifier.fillMaxSize().background(Color(0xFF0D0D0D))
            .padding(top = 52.dp, start = 16.dp, end = 16.dp, bottom = 24.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Text("broot", fontSize = 28.sp, fontWeight = FontWeight.Bold,
            color = Color.White, letterSpacing = 4.sp)
        Text("bar · root · analyzer", fontSize = 11.sp,
            color = Color(0xFF444444), letterSpacing = 2.sp)
        Spacer(Modifier.height(40.dp))
        when (s) {
            is AnalysisState.Idle      -> PickButton(::pickFile)
            is AnalysisState.Analyzing -> AnalyzingView(s.progress, s.status)
            is AnalysisState.Error     -> ErrorView(s.message) { state = AnalysisState.Idle }
            else -> {}
        }
    }
}

// ── DoneView ──────────────────────────────────────────────────────────────────

@Composable
fun DoneView(result: AnalysisResult, uri: Uri, fileName: String, onReset: () -> Unit) {
    val ctx   = LocalContext.current
    val scope = rememberCoroutineScope()

    val player      = remember { MediaPlayer() }
    var playerReady by remember { mutableStateOf(false) }
    var isPlaying   by remember { mutableStateOf(false) }
    var currentMs   by remember { mutableStateOf(0) }
    var totalMs     by remember { mutableStateOf(0) }
    var isSeeking   by remember { mutableStateOf(false) }
    var seekTarget  by remember { mutableStateOf(0) }

    var eq       by remember { mutableStateOf<Equalizer?>(null) }
    var eqBands  by remember { mutableStateOf<List<EqBandInfo>>(emptyList()) }
    var eqLevels by remember { mutableStateOf(IntArray(0)) }

    val bassEnabled  = remember { mutableStateListOf(*Array(result.bars.size) { true }) }
    var bassOctave   by remember { mutableStateOf(2) }   // 1–4
    var exporting    by remember { mutableStateOf(false) }
    var exportStatus by remember { mutableStateOf("") }

    var tab by remember { mutableStateOf(0) }

    LaunchedEffect(uri) {
        try {
            withContext(Dispatchers.IO) { player.setDataSource(ctx, uri); player.prepare() }
            totalMs = player.duration
            player.setOnCompletionListener { isPlaying = false; currentMs = 0 }
            playerReady = true
            try {
                val e = Equalizer(0, player.audioSessionId).also { it.enabled = true }
                val range = e.bandLevelRange
                eqBands  = (0 until e.numberOfBands.toInt()).map { b ->
                    EqBandInfo(b, e.getCenterFreq(b.toShort()) / 1000, range[0].toInt(), range[1].toInt())
                }
                eqLevels = IntArray(eqBands.size) { 0 }
                eq = e
            } catch (_: Exception) {}
        } catch (_: Exception) {}
    }

    LaunchedEffect(isPlaying) {
        while (isPlaying) {
            if (!isSeeking && playerReady) currentMs = player.currentPosition
            delay(80)
        }
    }

    DisposableEffect(Unit) { onDispose { eq?.release(); player.release() } }

    val barDurationMs = if (result.bpm > 0) ((60.0 / result.bpm) * 4 * 1000).toInt() else 1
    val currentBar    = (currentMs / barDurationMs).coerceIn(0, result.bars.size - 1)
    val displayMs     = if (isSeeking) seekTarget else currentMs
    val timelineState = rememberLazyListState()

    LaunchedEffect(currentBar) {
        if (isPlaying && !isSeeking) timelineState.animateScrollToItem(currentBar)
    }

    Column(
        modifier = Modifier
            .fillMaxWidth()
            .background(Color(0xFF0D0D0D))
            .verticalScroll(rememberScrollState())
            .padding(top = 48.dp, start = 16.dp, end = 16.dp, bottom = 32.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Text("broot", fontSize = 20.sp, fontWeight = FontWeight.Bold,
            color = Color.White, letterSpacing = 4.sp)
        Spacer(Modifier.height(16.dp))
        Row(horizontalArrangement = Arrangement.spacedBy(40.dp)) {
            InfoPill("BPM", "${result.bpm.toInt()}")
            InfoPill("BARS", "${result.bars.size}")
        }
        Spacer(Modifier.height(4.dp))
        Text(fileName.substringAfterLast("/"), fontSize = 10.sp,
            color = Color(0xFF333333), maxLines = 1)
        Spacer(Modifier.height(20.dp))

        // ── Player ──
        if (playerReady) {
            Box(
                contentAlignment = Alignment.Center,
                modifier = Modifier.size(52.dp)
                    .clip(RoundedCornerShape(26.dp))
                    .background(Color(0xFF6650A4))
                    .clickable {
                        if (isPlaying) { player.pause(); isPlaying = false }
                        else { player.start(); isPlaying = true }
                    }
            ) { Text(if (isPlaying) "⏸" else "▶", fontSize = 20.sp, color = Color.White) }

            Spacer(Modifier.height(10.dp))
            Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.fillMaxWidth()) {
                Text(formatMs(displayMs), fontSize = 11.sp, color = Color(0xFF555555),
                    modifier = Modifier.width(36.dp))
                Slider(
                    value = displayMs.toFloat(),
                    onValueChange = { isSeeking = true; seekTarget = it.toInt() },
                    onValueChangeFinished = {
                        currentMs = seekTarget
                        if (playerReady) {
                            if (Build.VERSION.SDK_INT >= 26)
                                player.seekTo(seekTarget.toLong(), MediaPlayer.SEEK_CLOSEST)
                            else player.seekTo(seekTarget)
                        }
                        isSeeking = false
                    },
                    valueRange = 0f..maxOf(totalMs.toFloat(), 1f),
                    modifier   = Modifier.weight(1f),
                    colors     = SliderDefaults.colors(
                        thumbColor         = Color(0xFF6650A4),
                        activeTrackColor   = Color(0xFF6650A4),
                        inactiveTrackColor = Color(0xFF2A2A2A)
                    )
                )
                Text(formatMs(totalMs), fontSize = 11.sp, color = Color(0xFF555555),
                    modifier = Modifier.width(36.dp), textAlign = TextAlign.End)
            }
        } else {
            CircularProgressIndicator(color = Color(0xFF6650A4), modifier = Modifier.size(30.dp))
            Spacer(Modifier.height(12.dp))
        }

        Spacer(Modifier.height(12.dp))
        Box(modifier = Modifier.fillMaxWidth().height(1.dp).background(Color(0xFF1E1E1E)))
        Spacer(Modifier.height(10.dp))

        // ── Tabs ──
        Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(4.dp)) {
            listOf("TIMELINE", "BASS", "EQ").forEachIndexed { i, label ->
                Box(
                    contentAlignment = Alignment.Center,
                    modifier = Modifier.weight(1f).height(32.dp)
                        .clip(RoundedCornerShape(8.dp))
                        .background(if (tab == i) Color(0xFF6650A4) else Color(0xFF1A1A1A))
                        .clickable { tab = i }
                ) {
                    Text(label, fontSize = 11.sp, fontWeight = FontWeight.SemiBold,
                        letterSpacing = 1.sp,
                        color = if (tab == i) Color.White else Color(0xFF555555))
                }
            }
        }
        Spacer(Modifier.height(14.dp))

        when (tab) {
            0 -> TimelineTab(result.bars, currentBar, isPlaying, timelineState)
            1 -> BassTab(
                    result      = result,
                    bassEnabled = bassEnabled,
                    octave      = bassOctave,
                    exporting   = exporting,
                    exportStatus= exportStatus,
                    onToggle    = { i -> bassEnabled[i] = !bassEnabled[i] },
                    onToggleAll = { v -> for (i in bassEnabled.indices) bassEnabled[i] = v },
                    onOctaveChange = { bassOctave = it },
                    onExport    = {
                        scope.launch {
                            exporting = true; exportStatus = ""
                            runCatching {
                                saveMix(ctx, uri, result, bassEnabled.toList(), bassOctave) {
                                    exportStatus = it
                                }
                            }.onSuccess { path -> exportStatus = "Saved:\n$path" }
                             .onFailure { exportStatus = "Error: ${it.message}" }
                            exporting = false
                        }
                    }
                )
            2 -> EqTab(eqBands, eqLevels) { band, level ->
                    eqLevels = eqLevels.copyOf().also { it[band] = level }
                    eq?.setBandLevel(band.toShort(), level.toShort())
                }
        }

        Spacer(Modifier.height(28.dp))
        TextButton(onClick = onReset) {
            Text("Analyze Another", color = Color(0xFF444444), letterSpacing = 1.sp)
        }
    }
}

// ── Timeline tab ──────────────────────────────────────────────────────────────

@Composable
fun TimelineTab(
    bars: List<BarResult>,
    currentBar: Int,
    isPlaying: Boolean,
    listState: LazyListState
) {
    LazyRow(
        state = listState,
        horizontalArrangement = Arrangement.spacedBy(6.dp),
        contentPadding = PaddingValues(horizontal = 4.dp)
    ) {
        itemsIndexed(bars) { i, bar ->
            BarCard(bar, isActive = isPlaying && i == currentBar)
        }
    }
}

// ── Bass tab ──────────────────────────────────────────────────────────────────

@Composable
fun BassTab(
    result: AnalysisResult,
    bassEnabled: List<Boolean>,
    octave: Int,
    exporting: Boolean,
    exportStatus: String,
    onToggle: (Int) -> Unit,
    onToggleAll: (Boolean) -> Unit,
    onOctaveChange: (Int) -> Unit,
    onExport: () -> Unit
) {
    Column(horizontalAlignment = Alignment.CenterHorizontally) {

        // ── Octave selector ──
        Text("O C T A V E", fontSize = 10.sp, color = Color(0xFF444444), letterSpacing = 3.sp,
            modifier = Modifier.align(Alignment.Start).padding(start = 4.dp, bottom = 8.dp))
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(6.dp)
        ) {
            (1..4).forEach { oct ->
                val selected = oct == octave
                val freqHint = when (oct) {
                    1 -> "~32 Hz"
                    2 -> "~65 Hz"
                    3 -> "~130 Hz"
                    else -> "~260 Hz"
                }
                Box(
                    contentAlignment = Alignment.Center,
                    modifier = Modifier.weight(1f).height(58.dp)
                        .clip(RoundedCornerShape(10.dp))
                        .background(if (selected) Color(0xFF26A69A) else Color(0xFF1A1A1A))
                        .clickable { onOctaveChange(oct) }
                ) {
                    Column(horizontalAlignment = Alignment.CenterHorizontally) {
                        Text(
                            OCTAVE_LABEL[oct] ?: "$oct",
                            fontSize = 13.sp, fontWeight = FontWeight.Bold,
                            color = if (selected) Color.White else Color(0xFF555555)
                        )
                        Text(
                            freqHint,
                            fontSize = 9.sp,
                            color = if (selected) Color(0xAAFFFFFF) else Color(0xFF333333)
                        )
                    }
                }
            }
        }

        Spacer(Modifier.height(16.dp))
        Box(modifier = Modifier.fillMaxWidth().height(1.dp).background(Color(0xFF1A1A1A)))
        Spacer(Modifier.height(12.dp))

        // ── Bar toggles ──
        Row(
            modifier = Modifier.fillMaxWidth().padding(horizontal = 4.dp),
            horizontalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            SmallChip("All On")  { onToggleAll(true) }
            SmallChip("All Off") { onToggleAll(false) }
        }
        Spacer(Modifier.height(10.dp))

        LazyRow(
            horizontalArrangement = Arrangement.spacedBy(6.dp),
            contentPadding = PaddingValues(horizontal = 4.dp)
        ) {
            itemsIndexed(result.bars) { i, bar ->
                BassBarCard(bar, enabled = bassEnabled[i]) { onToggle(i) }
            }
        }

        Spacer(Modifier.height(14.dp))
        val enabledCount = bassEnabled.count { it }
        Text("$enabledCount / ${result.bars.size} bars active",
            fontSize = 11.sp, color = Color(0xFF555555))
        Spacer(Modifier.height(14.dp))

        // ── Save button ──
        Box(
            contentAlignment = Alignment.Center,
            modifier = Modifier.fillMaxWidth(0.68f).height(44.dp)
                .clip(RoundedCornerShape(12.dp))
                .background(
                    if (!exporting && enabledCount > 0) Color(0xFF26A69A) else Color(0xFF1E1E1E)
                )
                .clickable(enabled = !exporting && enabledCount > 0) { onExport() }
        ) {
            Text(
                if (exporting) exportStatus.ifEmpty { "Working…" } else "Save Mix",
                fontSize = 14.sp, fontWeight = FontWeight.SemiBold,
                color = if (!exporting && enabledCount > 0) Color.White else Color(0xFF444444)
            )
        }

        if (!exporting && exportStatus.isNotEmpty()) {
            Spacer(Modifier.height(10.dp))
            Text(exportStatus, fontSize = 10.sp, color = Color(0xFF555555),
                textAlign = TextAlign.Center,
                modifier = Modifier.padding(horizontal = 8.dp))
        }
    }
}

@Composable
fun BassBarCard(bar: BarResult, enabled: Boolean, onClick: () -> Unit) {
    val color = if (enabled) noteColor(bar.rootNote) else Color(0xFF2A2A2A)
    Column(
        horizontalAlignment = Alignment.CenterHorizontally,
        modifier = Modifier.width(54.dp).clickable { onClick() }
    ) {
        Text("${bar.barIndex + 1}", fontSize = 10.sp, textAlign = TextAlign.Center,
            color = if (enabled) Color(0xFF888888) else Color(0xFF333333))
        Spacer(Modifier.height(4.dp))
        Box(
            contentAlignment = Alignment.Center,
            modifier = Modifier.fillMaxWidth().height(68.dp)
                .clip(RoundedCornerShape(10.dp))
                .background(color.copy(alpha = if (enabled) 0.88f else 0.22f))
        ) {
            Text(bar.rootNote, fontSize = 17.sp, fontWeight = FontWeight.Bold,
                color = if (enabled) Color.White else Color(0xFF444444))
        }
        Spacer(Modifier.height(4.dp))
        Box(modifier = Modifier.size(4.dp).clip(RoundedCornerShape(2.dp))
            .background(if (enabled) color.copy(alpha = 0.7f) else Color(0xFF222222)))
    }
}

@Composable
fun SmallChip(label: String, onClick: () -> Unit) {
    Box(
        contentAlignment = Alignment.Center,
        modifier = Modifier.height(28.dp)
            .clip(RoundedCornerShape(6.dp))
            .background(Color(0xFF1E1E1E))
            .clickable { onClick() }
            .padding(horizontal = 12.dp)
    ) { Text(label, fontSize = 11.sp, color = Color(0xFF888888)) }
}

// ── EQ tab ────────────────────────────────────────────────────────────────────

@Composable
fun EqTab(
    bands: List<EqBandInfo>,
    levels: IntArray,
    onLevelChange: (Int, Int) -> Unit
) {
    if (bands.isEmpty()) {
        Text("EQ unavailable on this device", fontSize = 13.sp, color = Color(0xFF555555))
        return
    }
    Column(
        modifier = Modifier.fillMaxWidth().padding(horizontal = 4.dp),
        verticalArrangement = Arrangement.spacedBy(6.dp)
    ) {
        bands.forEach { band ->
            val level = if (levels.size > band.band) levels[band.band] else 0
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text(formatFreq(band.centerFreqHz), fontSize = 11.sp, color = Color(0xFF666666),
                    modifier = Modifier.width(34.dp), textAlign = TextAlign.End)
                Slider(
                    value    = level.toFloat(),
                    onValueChange = { onLevelChange(band.band, it.toInt()) },
                    valueRange = band.minLevel.toFloat()..band.maxLevel.toFloat(),
                    modifier = Modifier.weight(1f).padding(horizontal = 8.dp),
                    colors   = SliderDefaults.colors(
                        thumbColor         = Color(0xFF6650A4),
                        activeTrackColor   = Color(0xFF6650A4),
                        inactiveTrackColor = Color(0xFF2A2A2A)
                    )
                )
                Text("%.1f dB".format(level / 100f), fontSize = 10.sp, color = Color(0xFF666666),
                    modifier = Modifier.width(52.dp))
            }
        }
    }
}

// ── BarCard ───────────────────────────────────────────────────────────────────

@Composable
fun BarCard(bar: BarResult, isActive: Boolean = false) {
    val color = noteColor(bar.rootNote)
    Column(horizontalAlignment = Alignment.CenterHorizontally, modifier = Modifier.width(54.dp)) {
        Text("${bar.barIndex + 1}", fontSize = 10.sp, textAlign = TextAlign.Center,
            color = if (isActive) Color.White else Color(0xFF444444))
        Spacer(Modifier.height(4.dp))
        Box(
            contentAlignment = Alignment.Center,
            modifier = Modifier.fillMaxWidth()
                .height(if (isActive) 76.dp else 68.dp)
                .clip(RoundedCornerShape(10.dp))
                .background(color.copy(alpha = if (isActive) 1f else 0.88f))
        ) {
            Text(bar.rootNote, fontWeight = FontWeight.Bold, color = Color.White,
                fontSize = if (isActive) 19.sp else 17.sp)
        }
        Spacer(Modifier.height(4.dp))
        Box(modifier = Modifier
            .size(if (isActive) 6.dp else 4.dp)
            .clip(RoundedCornerShape(3.dp))
            .background(color.copy(alpha = bar.confidence.coerceIn(0.15f, 1f))))
    }
}

// ── Shared ────────────────────────────────────────────────────────────────────

@Composable
fun PickButton(onClick: () -> Unit) {
    Box(
        contentAlignment = Alignment.Center,
        modifier = Modifier.fillMaxWidth(0.72f).height(54.dp)
            .clip(RoundedCornerShape(14.dp))
            .background(Color(0xFF6650A4))
            .clickable { onClick() }
    ) { Text("Pick Audio File", fontSize = 16.sp, fontWeight = FontWeight.SemiBold, color = Color.White) }
}

@Composable
fun AnalyzingView(progress: Float, status: String) {
    Column(horizontalAlignment = Alignment.CenterHorizontally) {
        LinearProgressIndicator(
            progress   = { progress },
            modifier   = Modifier.fillMaxWidth().height(6.dp).clip(RoundedCornerShape(3.dp)),
            color      = Color(0xFF6650A4),
            trackColor = Color(0xFF2A2A2A)
        )
        Spacer(Modifier.height(14.dp))
        Text(status, fontSize = 14.sp, color = Color(0xFF999999))
        Spacer(Modifier.height(4.dp))
        Text("${(progress * 100).toInt()}%", fontSize = 11.sp, color = Color(0xFF555555))
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
        Text("Analysis failed", fontSize = 16.sp, color = Color(0xFFEF5350),
            fontWeight = FontWeight.SemiBold)
        Spacer(Modifier.height(10.dp))
        Text(message, fontSize = 13.sp, color = Color(0xFF777777), textAlign = TextAlign.Center)
        Spacer(Modifier.height(22.dp))
        TextButton(onClick = onReset) { Text("Try Again", color = Color(0xFF6650A4)) }
    }
}
