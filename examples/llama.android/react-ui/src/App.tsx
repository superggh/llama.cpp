import { useState, useEffect, useCallback, useRef } from 'react'

declare global {
  interface Window {
    Android?: {
      updateConfig: (key: string, value: string) => void
      startVoice: () => void
      stopVoice: () => void
      getConfig: () => string
      getAudioLevel: () => number
      setTtsEnabled: (enabled: boolean) => void
    }
  }
}

type Lang = 'Auto' | 'Chinese' | 'English' | 'Japanese' | 'Korean' | 'Spanish' | 'French' | 'German'
type State = 'Idle' | 'Ready' | 'Listening' | 'Transcribing' | 'Translating' | 'Error'

interface Config {
  silenceThresholdDb: number
  minSpeechDurationMs: number
  minSilenceDurationMs: number
  forceSplitSec: number
  sourceLanguage: Lang
  targetLanguage: Lang
  encoderThreads: number
  partialIntervalMs: number
  ttsEnabled: boolean
}

interface Timing {
  fbank: number
  encoder: number
  llm_decode: number
  sample: number
}

interface Message {
  text: string
  translated: string
  isUser: boolean
}

const LANGUAGES: Lang[] = ['Auto', 'Chinese', 'English', 'Japanese', 'Korean', 'Spanish', 'French', 'German']

function App() {
  const [bridgeReady, setBridgeReady] = useState(false)
  const [state, setState] = useState<State>('Idle')
  const [config, setConfig] = useState<Config>({
    silenceThresholdDb: 28, minSpeechDurationMs: 400, minSilenceDurationMs: 400,
    forceSplitSec: 4, sourceLanguage: 'Auto', targetLanguage: 'English',
    encoderThreads: 4, partialIntervalMs: 4000, ttsEnabled: false,
  })
  const [timing, setTiming] = useState<Timing | null>(null)
  const [messages, setMessages] = useState<Message[]>([])
  const [audioLevel, setAudioLevel] = useState(-50)
  const msgEndRef = useRef<HTMLDivElement>(null)

  useEffect(() => { msgEndRef.current?.scrollIntoView({ behavior: 'smooth' }) }, [messages])

  useEffect(() => {
    const check = setInterval(() => {
      if (window.Android) {
        setBridgeReady(true)
        clearInterval(check)
      }
    }, 100)
    return () => clearInterval(check)
  }, [])

  // Poll audio level every 200ms
  useEffect(() => {
    if (!bridgeReady) return
    const interval = setInterval(() => {
      const level = window.Android?.getAudioLevel()
      if (level !== undefined) setAudioLevel(level)
    }, 200)
    return () => clearInterval(interval)
  }, [bridgeReady])

  // Listen for native events
  useEffect(() => {
    const handler = (e: any) => {
      const { type, data } = e.detail || {}
      if (type === 'state') { setState(data) }
      else if (type === 'timing') { setTiming(data) }
      else if (type === 'transcribed') { setMessages(p => [...p, { text: data, translated: '', isUser: true }]) }
      else if (type === 'translated') { setMessages(p => { const m = [...p]; if (m.length) m[m.length - 1].translated = data; return m }) }
      else if (type === 'status') { console.log('status:', data) }
    }
    window.addEventListener('native-event', handler as any)
    return () => window.removeEventListener('native-event', handler as any)
  }, [])

  const sendConfig = useCallback((key: string, value: any) => {
    window.Android?.updateConfig(key, String(value))
    setConfig(c => ({ ...c, [key]: value }))
  }, [])

  const toggleVoice = () => {
    if (state === 'Listening') window.Android?.stopVoice()
    else window.Android?.startVoice()
  }

  if (!bridgeReady) {
    return (
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100vh', background: '#0f172a', color: '#64748b', fontSize: 18 }}>
        Connecting...
      </div>
    )
  }

  const statusColor = state === 'Listening' ? '#4ade80' : state === 'Transcribing' || state === 'Translating' ? '#facc15' : '#94a3b8'
  const stateLabels: Record<State, string> = { Idle: 'Idle', Ready: 'Ready', Listening: 'Listening…', Transcribing: 'Transcribing…', Translating: 'Translating…', Error: 'Error' }

  return (
    <div style={{ padding: 16, fontFamily: 'system-ui, sans-serif', background: '#0f172a', color: '#e2e8f0', minHeight: '100vh' }}>
      {/* Status bar */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 16 }}>
        <div style={{ width: 12, height: 12, borderRadius: 6, background: statusColor }} />
        <span style={{ fontSize: 18, fontWeight: 600 }}>{stateLabels[state]}</span>
        {timing && <span style={{ fontSize: 12, color: '#94a3b8', marginLeft: 'auto' }}>
          enc {timing.encoder}ms · llm {timing.llm_decode}ms · smp {timing.sample}ms
        </span>}
      </div>

      {/* Language selectors */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr auto 1fr', gap: 8, marginBottom: 16, alignItems: 'center' }}>
        <select value={config.sourceLanguage} onChange={e => sendConfig('sourceLanguage', e.target.value)} style={selectStyle}>
          {LANGUAGES.map(l => <option key={l}>{l}</option>)}
        </select>
        <span style={{ fontSize: 20, color: '#64748b' }}>→</span>
        <select value={config.targetLanguage} onChange={e => sendConfig('targetLanguage', e.target.value)} style={selectStyle}>
          {LANGUAGES.filter(l => l !== 'Auto').map(l => <option key={l}>{l}</option>)}
        </select>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <button onClick={() => {
            const val = !config.ttsEnabled
            sendConfig('ttsEnabled', val)
            window.Android?.setTtsEnabled(val)
          }} style={{
            ...btnStyle, padding: '6px 12px', fontSize: 13,
            background: config.ttsEnabled ? '#059669' : '#374151',
          }}>TTS {config.ttsEnabled ? 'ON' : 'OFF'}</button>
        </div>
      </div>

      {/* VAD sliders */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px 16px', marginBottom: 8, padding: 12, background: '#1e293b', borderRadius: 8 }}>
        <Slider label="Threshold (dB)" value={config.silenceThresholdDb} min={20} max={50} step={0.5} onChange={v => sendConfig('silenceThresholdDb', v)} />
        <Slider label="Speech (ms)" value={config.minSpeechDurationMs} min={100} max={1000} step={50} onChange={v => sendConfig('minSpeechDurationMs', v)} />
        <Slider label="Silence (ms)" value={config.minSilenceDurationMs} min={200} max={2000} step={50} onChange={v => sendConfig('minSilenceDurationMs', v)} />
        <Slider label="Force Split (s)" value={config.forceSplitSec} min={2} max={15} step={1} onChange={v => sendConfig('forceSplitSec', v)} />
      </div>

      {/* Audio level meter */}
      <LevelMeter level={audioLevel} threshold={config.silenceThresholdDb} />

      {/* Voice button */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 16 }}>
        <button onClick={toggleVoice} style={{
          ...btnStyle, flex: 1, background: state === 'Listening' ? '#dc2626' : '#2563eb',
          padding: '14px 0', fontSize: 16,
        }}>{state === 'Listening' ? '■ Stop Voice' : '● Start Voice'}</button>
      </div>

      {/* Messages */}
      <div style={{ background: '#1e293b', borderRadius: 8, padding: 12, minHeight: 200, fontSize: 14 }}>
        {messages.length === 0 && <div style={{ color: '#64748b', textAlign: 'center', paddingTop: 40 }}>Transcriptions appear here</div>}
        {messages.map((m, i) => (
          <div key={i} style={{ marginBottom: 8 }}>
            <div style={{ color: '#38bdf8' }}>{m.text}</div>
            {m.translated && <div style={{ color: '#a78bfa', marginTop: 2 }}>{m.translated}</div>}
          </div>
        ))}
        <div ref={msgEndRef} />
      </div>
    </div>
  )
}

function LevelMeter({ level, threshold }: { level: number; threshold: number }) {
  const pct = Math.max(0, Math.min(100, ((level - 20) / (50 - 20)) * 100))
  const threshPct = Math.max(0, Math.min(100, ((threshold - 20) / (50 - 20)) * 100))
  const barColor = level > threshold ? '#ef4444' : '#3b82f6'
  return (
    <div style={{ marginBottom: 16, padding: '8px 12px', background: '#1e293b', borderRadius: 8 }}>
      <div style={{ fontSize: 12, color: '#94a3b8', marginBottom: 4 }}>Audio Level: {level.toFixed(1)} dB</div>
      <div style={{ position: 'relative', height: 12, background: '#0f172a', borderRadius: 6, overflow: 'hidden' }}>
        <div style={{ width: `${pct}%`, height: '100%', background: barColor, borderRadius: 6, transition: 'width 100ms' }} />
        <div style={{ position: 'absolute', top: 0, left: `${threshPct}%`, width: 2, height: '100%', background: '#facc15' }} />
      </div>
    </div>
  )
}

function Slider({ label, value, min, max, step, onChange }: { label: string; value: number; min: number; max: number; step: number; onChange: (v: number) => void }) {
  return (
    <div>
      <div style={{ fontSize: 12, color: '#94a3b8', marginBottom: 2 }}>{label}: {value}</div>
      <input type="range" min={min} max={max} step={step} value={value}
        onChange={e => onChange(Number(e.target.value))}
        style={{ width: '100%', accentColor: '#3b82f6' }} />
    </div>
  )
}

const selectStyle: React.CSSProperties = {
  width: '100%', padding: '10px 4px', borderRadius: 6, border: '1px solid #334155',
  background: '#1e293b', color: '#e2e8f0', fontSize: 16,
}

const btnStyle: React.CSSProperties = {
  borderRadius: 10, border: 'none', color: '#fff',
  fontWeight: 700, cursor: 'pointer',
}

export default App
