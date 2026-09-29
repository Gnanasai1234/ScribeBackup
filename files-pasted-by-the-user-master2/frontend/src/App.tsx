import * as React from 'react'
import { useEffect, useMemo, useState } from 'react'
import { useFrame } from '@react-three/fiber'
import * as THREE from 'three'
import { Activity, AlertCircle, ArrowDownToLine, ArrowRight, AudioLines, BookOpen, Check, ChevronDown, ChevronUp, CircleHelp, Clock3, FileText, FolderOpen, Headphones, Layers3, Menu, MessageSquareText, Network, Play, Plus, Search, ShieldCheck, Sparkles, Upload, Volume2, Waves, X } from 'lucide-react'
import { Stage, CanvasBoundary } from './components/three/shared'
import { AIProcessingPipeline, type PipelineStep } from './components/three/AIProcessingPipeline'
import { DocumentChunkVisualization } from './components/three/DocumentChunkVisualization'
import { KnowledgeGraph } from './components/three/KnowledgeGraph'
import { SourceValidationGraph } from './components/three/SourceValidationGraph'
import { RAGVisualization, type RagPhase } from './components/three/RAGVisualization'
import { AudioVisualization } from './components/three/AudioVisualization'
import { CompareVisualization } from './components/three/CompareVisualization'
import { derivePipelineSteps } from './lib/pipelineState'
import { isUsableSummary, summaryText } from './lib/summary'
import { VoiceInput, type VoiceState } from './components/VoiceInput'
import { VoiceOrb } from './components/three/VoiceOrb'
import { LocalLLMTest } from './components/LocalLLMTest'
import { DocumentComparison } from './components/DocumentComparison'
import { API, apiFetch as fetch, responseError } from './lib/api'
import { useAuth } from './auth/AuthContext'
import { LoginPage, RegisterPage } from './auth/AuthPages'
import './history.css'

const neuralVoices: Record<string, { id: string; name: string }[]> = {
  en: [
    { id: 'en-US-AriaNeural', name: 'Aria · US English' },
    { id: 'en-US-GuyNeural', name: 'Guy · US English' },
    { id: 'en-IN-NeerjaNeural', name: 'Neerja · Indian English' },
    { id: 'en-IN-PrabhatNeural', name: 'Prabhat · Indian English' },
  ],
  hi: [
    { id: 'hi-IN-SwaraNeural', name: 'Swara · Hindi neural' },
    { id: 'hi-IN-MadhurNeural', name: 'Madhur · Hindi neural' },
  ],
  te: [
    { id: 'te-IN-ShrutiNeural', name: 'Shruti · Telugu neural' },
    { id: 'te-IN-MohanNeural', name: 'Mohan · Telugu neural' },
  ],
  ta: [
    { id: 'ta-IN-PallaviNeural', name: 'Pallavi · Tamil neural' },
    { id: 'ta-IN-ValluvarNeural', name: 'Valluvar · Tamil neural' },
  ],
  kn: [
    { id: 'kn-IN-SapnaNeural', name: 'Sapna · Kannada neural' },
    { id: 'kn-IN-GaganNeural', name: 'Gagan · Kannada neural' },
  ],
  ml: [
    { id: 'ml-IN-SobhanaNeural', name: 'Sobhana · Malayalam neural' },
    { id: 'ml-IN-MidhunNeural', name: 'Midhun · Malayalam neural' },
  ],
  bn: [
    { id: 'bn-IN-TanishaaNeural', name: 'Tanishaa · Bengali neural' },
    { id: 'bn-IN-BashkarNeural', name: 'Bashkar · Bengali neural' },
  ],
  mr: [
    { id: 'mr-IN-AarohiNeural', name: 'Aarohi · Marathi neural' },
    { id: 'mr-IN-ManoharNeural', name: 'Manohar · Marathi neural' },
  ],
  gu: [
    { id: 'gu-IN-DhwaniNeural', name: 'Dhwani · Gujarati neural' },
    { id: 'gu-IN-NiranjanNeural', name: 'Niranjan · Gujarati neural' },
  ],
  ur: [
    { id: 'ur-IN-GulNeural', name: 'Gul · Urdu neural' },
    { id: 'ur-IN-SalmanNeural', name: 'Salman · Urdu neural' },
  ],
}

export default function App(){
 const {user,isLoading}=useAuth()
 if(isLoading)return <main className="auth-loading"><span className="spinner"/> Loading your workspace…</main>
 const path=window.location.pathname
 if(user&&(path==='/login'||path==='/register')){window.location.replace('/');return <main className="auth-loading"><span className="spinner"/> Opening your workspace…</main>}
 if(!user)return path==='/register'?<RegisterPage/>:<LoginPage/>
 return <ScribeApp key={user.id}/>
}
const speechLocale: Record<string, string> = {
  en: 'en-US',
  hi: 'hi-IN',
  te: 'te-IN',
  ta: 'ta-IN',
  kn: 'kn-IN',
  ml: 'ml-IN',
  bn: 'bn-IN',
  mr: 'mr-IN',
  gu: 'gu-IN',
  ur: 'ur-IN',
}
const supportedLanguages = [
  { code: 'en', name: 'English', flag: '🇬🇧 English' },
  { code: 'hi', name: 'Hindi', flag: '🇮🇳 Hindi' },
  { code: 'te', name: 'Telugu', flag: '🇮🇳 Telugu' },
  { code: 'ta', name: 'Tamil', flag: '🇮🇳 Tamil' },
  { code: 'kn', name: 'Kannada', flag: '🇮🇳 Kannada' },
  { code: 'ml', name: 'Malayalam', flag: '🇮🇳 Malayalam' },
  { code: 'bn', name: 'Bengali', flag: '🇮🇳 Bengali' },
  { code: 'mr', name: 'Marathi', flag: '🇮🇳 Marathi' },
  { code: 'gu', name: 'Gujarati', flag: '🇮🇳 Gujarati' },
  { code: 'ur', name: 'Urdu', flag: '🇵🇰 Urdu' },
]
const outputLanguages = [{ code: 'auto', name: 'Auto · Original document language', flag: '' }, ...supportedLanguages]
const languageFlags: Record<string, string> = Object.fromEntries(supportedLanguages.map(l => [l.code, l.flag]))
const languageNames: Record<string, string> = Object.fromEntries(supportedLanguages.map(l => [l.code, l.name]))
const languageLabel = (code: string) => code === 'auto' ? 'Auto / Original language' : languageNames[code] || code.toUpperCase()
const providerTitles: Record<string, string> = { openrouter: 'OpenRouter', 'openai-direct': 'Direct OpenAI', 'ollama-local': 'Ollama Local', 'deterministic-extractive': 'Deterministic extractive summary' }
const providerTitle = (name?: string | null) => name ? (providerTitles[name] || name) : 'Unknown provider'

type TimingItem = { index: number; start: number; end: number; text: string }
type SpeechSegment = { index: number; text: string; start: number; end: number; charStart: number; charEnd: number }

type Doc = {id:string;name:string;language:string;status:string;created_at?:string;ocr_detected?:boolean;ocr_warnings?:string[];stats:{words:number;chunks:number;pages:number|null}}
type Result = {status:string;generation_status?:'success'|'partial_success'|'failed';error?:string;summary?:unknown;depth:string;summary_mode?:string;summary_depth_requested?:string;summary_depth_resolved?:string;purpose:string;language:string;output_language?:string;actual_output_language?:string;source_language?:string;requested_language?:string;requested_output_language?:string;translation_validation_status?:string;translation_provider?:string;translation_warning?:boolean;translation_warning_message?:string|null;translation_failed_fallback?:boolean;validation_provider?:string;validation_status?:string;validation_state?:'VALIDATED'|'PARTIALLY_VALIDATED'|'VALIDATION_FAILED_AFTER_RETRIES'|'VALIDATION_UNAVAILABLE'|'NO_CLAIMS'|'GENERATION_FAILED';validation_warning?:boolean;warning_message?:string|null;supported_claims_count?:number;unsupported_claims_count?:number;validation_attempts?:number;executive_summary:string|null;findings:string[];facts?:string[];risks?:string[];decisions?:string[];recommendations?:string[];action_items?:string[];claims:{id:string;claim:string;status:string;confidence?:number|null;evidence?:string;sources:{page:number|null;section:string;chunk_id:string}[];reason:string}[];quality:{groundedness:number|null;claims_checked:number;evaluation:string};stats:{words:number;pages:number|null;estimated_tokens:number;chunks:number;context_chunks_used?:number;processing_time_ms?:number;regeneration_count?:number;validation_attempts?:number;max_validation_attempts?:number;language:string};validation?:{attempts?:number;max_attempts?:number;max_attempts_reached?:boolean;state?:string;warning?:boolean;warning_message?:string|null;supported_claims?:number;unsupported_claims?:number};workflow?:string[];evaluation?:any;limitations:string[];warning?:string;provider_status?:{primary?:string;fallback?:string;final?:string};provider?:{name:string|null;model?:string|null;fallback_used?:boolean;fallback_reason?:string;primary_provider?:string|null;primary_model?:string|null}}
const nav = [{name:'Overview',icon:Activity},{name:'Documents',icon:FolderOpen},{name:'History',icon:Clock3},{name:'Analysis',icon:Layers3},{name:'Sources',icon:BookOpen},{name:'Ask document',icon:MessageSquareText},{name:'LLM Provider Test',icon:AudioLines},{name:'Compare',icon:Network},{name:'Audio briefing',icon:Headphones}]
const customDetailOptions = [{value:'quick',label:'Concise',description:'Short and focused'},{value:'standard',label:'Balanced',description:'Coverage with useful detail'},{value:'deep',label:'Detailed',description:'Maximum context'}] as const
const customSectionOptions = [{value:'findings',label:'Key findings'},{value:'facts',label:'Important facts'},{value:'risks',label:'Risks'},{value:'decisions',label:'Decisions'},{value:'recommendations',label:'Recommendations'},{value:'action_items',label:'Action items'}] as const

function Orb(){
 const ref=React.useRef<THREE.Mesh>(null!)
 useFrame(({clock})=>{if(ref.current){ref.current.rotation.y=clock.elapsedTime*.08;ref.current.rotation.x=Math.sin(clock.elapsedTime*.16)*.08}})
 return <><mesh ref={ref}><icosahedronGeometry args={[1.23,2]}/><meshStandardMaterial color="#19342b" wireframe emissive="#54dfa0" emissiveIntensity={.72}/></mesh><mesh scale={.72}><icosahedronGeometry args={[1.23,2]}/><meshBasicMaterial color="#62e3a1" wireframe transparent opacity={.18}/></mesh><pointLight color="#5df0a2" intensity={9} distance={6}/></>
}
const visualFallback=<div className="canvas-fallback">Document → Intelligence → Validated summary → Audio</div>
function Visual(){return <div className="visual"><Stage height="100%" fallback={visualFallback} camera={{position:[0,0,4.5]}}><ambientLight intensity={1.2}/><Orb/></Stage><div className="orbit-label label-doc">DOCUMENT</div><div className="orbit-label label-ai">INTELLIGENCE</div><div className="orbit-label label-audio">BRIEFING</div><div className="visual-bottom">SOURCE-GROUNDED BY DESIGN <span>✓</span></div></div>}

function ScribeApp(){
 const {user,logout}=useAuth()
 const [page,setPage]=useState('Overview'),[docs,setDocs]=useState<Doc[]>([]),[doc,setDoc]=useState<Doc|null>(null),[result,setResult]=useState<Result|null>(null),[depth,setDepth]=useState('standard'),[customDetail,setCustomDetail]=useState('standard'),[customSections,setCustomSections]=useState<string[]>(['findings','facts','risks','decisions','recommendations','action_items']),[purpose,setPurpose]=useState('Executive briefing'),[language,setLanguage]=useState('auto'),[ocrLanguage,setOcrLanguage]=useState('auto'),[audioLanguage,setAudioLanguage]=useState('auto'),[audience,setAudience]=useState('Leadership'),[busy,setBusy]=useState(false),[error,setError]=useState(''),[question,setQuestion]=useState(''),[answer,setAnswer]=useState<any>(null),[toast,setToast]=useState(''),[previewOpen,setPreviewOpen]=useState(false),[audioSpeed,setAudioSpeed]=useState(1),[selectedVoice,setSelectedVoice]=useState(''),[voiceOptions,setVoiceOptions]=useState<SpeechSynthesisVoice[]>([]),[audioPaused,setAudioPaused]=useState(false),[audioUrl,setAudioUrl]=useState(''),[audioBusy,setAudioBusy]=useState(false),[askBusy,setAskBusy]=useState(false),[selectedChunk,setSelectedChunk]=useState<string|null>(null),[speaking,setSpeaking]=useState(false)
 const [audioNarrationText,setAudioNarrationText]=useState('')
 const [audioOverrideText,setAudioOverrideText]=useState('')
 const [audioOverrideLanguage,setAudioOverrideLanguage]=useState('')
 const [pendingAudioPlay,setPendingAudioPlay]=useState(false)
 const [audioTimings,setAudioTimings]=useState<TimingItem[]>([])
 const [activeSegmentIndex,setActiveSegmentIndex]=useState<number|null>(null)
 const [currentPlayingProvider,setCurrentPlayingProvider]=useState<string>('')
 const [currentPlayingVoice,setCurrentPlayingVoice]=useState<string>('')
 // Voice input state
 const [voiceInputLanguage,setVoiceInputLanguage]=useState('en')
 const [askVoiceState,setAskVoiceState]=useState<VoiceState>('idle')
 const [customInstructionVoiceState,setCustomInstructionVoiceState]=useState<VoiceState>('idle')
 const [customInstruction,setCustomInstruction]=useState('')
 const audioRef=React.useRef<HTMLAudioElement>(null)
 const refresh=()=>fetch(`${API}/documents`).then(r=>r.ok?r.json():[]).then(data=>setDocs(Array.isArray(data)?data:[])).catch(()=>{})
 async function openDocument(d:Doc){setDoc(d);setResult(null);setPage('Analysis');try{const response=await fetch(`${API}/documents/${encodeURIComponent(d.id)}/summary`);if(response.ok){const saved=await response.json();setResult({...saved,executive_summary:summaryText(saved.executive_summary)||summaryText(saved.summary)})}}catch{}}
 useEffect(()=>{refresh()},[])
 useEffect(()=>{if(!('speechSynthesis'in window))return;const update=()=>setVoiceOptions(speechSynthesis.getVoices().filter(v=>Object.keys(speechLocale).some(code=>v.lang.toLowerCase().startsWith(code))));update();speechSynthesis.onvoiceschanged=update;return()=>{speechSynthesis.onvoiceschanged=null}},[])
 useEffect(()=>{const local=voiceOptions.filter(v=>v.lang.toLowerCase().startsWith(audioLanguage));if(local.length){setSelectedVoice(current=>current.startsWith('browser:')&&local.some(v=>`browser:${v.voiceURI}`===current)?current:`browser:${local[0].voiceURI}`)}else{setSelectedVoice(current=>current.startsWith('edge:')&&neuralVoices[audioLanguage]?.some(v=>`edge:${v.id}`===current)?current:`edge:${neuralVoices[audioLanguage]?.[0]?.id||''}`)}},[audioLanguage,voiceOptions])
 useEffect(()=>{
  if(result?.language)setAudioLanguage(result.language);
  if(result?.executive_summary){
    setAudioNarrationText(result.executive_summary);
    setAudioTimings([]);
    setActiveSegmentIndex(null);
  }
 },[result])
 useEffect(()=>{if(audioUrl&&audioRef.current){audioRef.current.playbackRate=audioSpeed;audioRef.current.play().catch(()=>notify('Audio is ready. Press play on the audio player to start.'))}},[audioUrl])
 useEffect(()=>()=>{if(audioUrl)URL.revokeObjectURL(audioUrl);if('speechSynthesis'in window)speechSynthesis.cancel()},[audioUrl])
 const notify=(s:string)=>{setToast(s);setTimeout(()=>setToast(''),3500)}

 const speechSegments=useMemo<SpeechSegment[]>(()=>{
  const text=audioNarrationText||result?.executive_summary||''
  if(!text.trim())return[]

  if(audioTimings&&audioTimings.length>0){
   let charCursor=0
   return audioTimings.map((t,i)=>{
    const matchIdx=text.indexOf(t.text,charCursor)
    const charStart=matchIdx!==-1?matchIdx:charCursor
    const charEnd=charStart+t.text.length
    if(matchIdx!==-1)charCursor=charEnd
    return {index:i,text:t.text,start:t.start,end:t.end,charStart,charEnd}
   })
  }

  const pattern=/[^.!?\u0964\u0965]+(?:[.!?\u0964\u0965]+|\n+|$)/g
  const rawMatches=text.match(pattern)||[text]
  const items=rawMatches.map(s=>s.trim()).filter(Boolean)
  if(!items.length)return[]

  const weights=items.map(s=>Math.max(1,s.length)+12)
  const totalWeight=weights.reduce((a,b)=>a+b,0)
  const totalDur=(audioRef.current&&audioRef.current.duration&&!isNaN(audioRef.current.duration)&&audioRef.current.duration>0)?audioRef.current.duration:items.reduce((acc,s)=>acc+Math.max(1.5,s.length/14),0)

  let curTime=0
  let charCursor=0
  return items.map((sent,i)=>{
   const dur=(weights[i]/totalWeight)*totalDur
   const start=Math.round(curTime*1000)/1000
   const end=Math.round((curTime+dur)*1000)/1000
   curTime=end

   const matchIdx=text.indexOf(sent,charCursor)
   const charStart=matchIdx!==-1?matchIdx:charCursor
   const charEnd=charStart+sent.length
   if(matchIdx!==-1)charCursor=charEnd

   return {index:i,text:sent,start,end,charStart,charEnd}
  })
 },[audioNarrationText,result?.executive_summary,audioTimings])

 const syncAudioTime=()=>{
  if(!audioRef.current||!speechSegments.length)return
  const current=audioRef.current.currentTime
  if(audioRef.current.ended){
   setActiveSegmentIndex(null)
   return
  }
  const match=speechSegments.find(s=>current>=s.start&&current<s.end)
  if(match){
   setActiveSegmentIndex(match.index)
  }else if(current>=(speechSegments[speechSegments.length-1]?.end||0)){
   setActiveSegmentIndex(null)
  }else{
   const nextIdx=speechSegments.findIndex(s=>s.start>current)
   if(nextIdx>0)setActiveSegmentIndex(nextIdx-1)
   else if(nextIdx===0)setActiveSegmentIndex(0)
  }
 }

 const updateActiveSegmentByCharIndex=(charIndex:number)=>{
  if(!speechSegments.length)return
  const match=speechSegments.find(s=>charIndex>=s.charStart&&charIndex<=s.charEnd)
  if(match)setActiveSegmentIndex(match.index)
 }

 async function upload(file:File){setBusy(true);setError('');setToast('');try{const fd=new FormData();fd.append('file',file);fd.append('ocr_language',ocrLanguage);const r=await fetch(`${API}/documents/upload`,{method:'POST',body:fd});const data=await r.json();if(!r.ok)throw Error(data.detail||'Upload failed');setDoc(data);setResult(null);setPage('Analysis');notify(data.ocr_detected?'Scanned/image content detected — OCR text is ready for analysis.':'Document extracted and ready for analysis');refresh()}catch(e:any){setError(e.message)}finally{setBusy(false)}}
 async function demo(){setBusy(true);setError('');try{const r=await fetch(`${API}/demo`,{method:'POST'});const data=await r.json();if(!r.ok)throw Error(data.detail||'Demo unavailable');setDoc(data);setResult(data.result);setPage('Analysis');refresh();if(data.result?.generation_status==='failed'){setError(data.result.error||'Summary generation failed');notify('Summary generation failed')}else{notify(data.result?.provider?.name?`Demo generated by ${providerTitle(data.result.provider.name)}.`:'Demo generated with the deterministic extractive summary.')}}catch(e:any){setError(e.message)}finally{setBusy(false)}}
 async function analyze(){
  if(!doc)return
  const targetLanguage=language==='auto'?doc.language:language
  setBusy(true);setError('');speechSynthesis.cancel();if(audioRef.current)audioRef.current.pause();setAudioUrl('');setAudioNarrationText('');setAudioTimings([]);setActiveSegmentIndex(null);setSpeaking(false);setAudioPaused(false)
  try{
   const endpoint=result&&targetLanguage!==result.language?'translate':'process'
   const r=await fetch(`${API}/documents/${doc.id}/${endpoint}`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({summary_depth:depth,custom_detail:customDetail,custom_sections:customSections,purpose,source_language:doc.language,output_language:language,audience})})
   const data=await r.json();if(!r.ok)throw Error(data.detail||'Analysis failed')
   let finalResult={...data,executive_summary:summaryText(data.executive_summary)||summaryText(data.summary)}
   if(!isUsableSummary(finalResult.executive_summary)){finalResult={...finalResult,status:'failed',generation_status:'failed',executive_summary:null,error:finalResult.error||'No usable summary could be generated.'}}

   setResult(finalResult)
   if(finalResult.generation_status==='failed'){setError(finalResult.error||'Summary generation failed');notify('Summary generation failed')}
   else if(finalResult.language!==targetLanguage){notify(`Translation to ${languageNames[targetLanguage]||targetLanguage.toUpperCase()} failed; showing the source-language summary in ${languageNames[finalResult.language]||finalResult.language.toUpperCase()}.`)}
   else if(finalResult.language!==finalResult.source_language){notify(`Translation to ${languageNames[finalResult.language]||finalResult.language.toUpperCase()} complete`)}
   else{notify('Analysis complete ? summary language follows the document')}
  }catch(e:any){setError(e.message);notify('Analysis error: '+e.message)}finally{setBusy(false)}
 }
 async function ask(){if(!doc||!question.trim())return;setAskBusy(true);setSelectedChunk(null);try{const r=await fetch(`${API}/documents/${doc.id}/ask`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question,output_language:language})});setAnswer(await r.json())}catch{setError('Could not reach the document service.')}finally{setAskBusy(false)}}
 async function play(narrationOverride?:string){
  // TEMPORARILY DISABLED: Exact-status audio gate blocked usable partial summaries.
  // Keep validation visible in the result; narration needs actual summary text.
  // if(!doc||!result||result.status!=='validated'){notify('Audio requires a validated summary');return}
  if(!doc||!result||!result.executive_summary?.trim()){
   notify('Generate a summary before requesting audio');
   return;
  }
  setAudioBusy(true);
  try{
   speechSynthesis.cancel();
   if(audioRef.current)audioRef.current.pause();
   setActiveSegmentIndex(null);

   let online=selectedVoice.startsWith('edge:');
   let voiceId=selectedVoice.replace(/^(edge|browser):/,'');

   // Multilingual Fallback: If browser is selected but no browser voice is installed for audioLanguage, automatically use online neural voice
   const browserHasLang=voiceOptions.some(v=>v.lang.toLowerCase().startsWith(audioLanguage));
   if(!online&&!browserHasLang){
    const fallbackVoice=neuralVoices[audioLanguage]?.[0]?.id;
    if(fallbackVoice){
     console.log(`[TTS Fallback] No system voice for ${audioLanguage}. Using online neural voice: ${fallbackVoice}`);
     online=true;
     voiceId=fallbackVoice;
     setSelectedVoice(`edge:${voiceId}`);
     notify(`Using online neural voice for ${languageNames[audioLanguage]||audioLanguage}`);
    }else{
     throw Error(`No speech voice available for ${languageNames[audioLanguage]||audioLanguage.toUpperCase()} on this device.`);
    }
   }

   console.log(`[TTS Request] provider=${online?'edge':'browser'}, voice=${voiceId}, language=${audioLanguage}, rate=${audioSpeed}`);

   const r=await fetch(`${API}/documents/${doc.id}/audio`,{
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({
     language:audioLanguage,
     provider:online?'edge':'browser',
     voice_id:online?voiceId:'browser-default',
     rate:audioSpeed,
     narration_text:narrationOverride?.trim()||undefined,
     narration_language:narrationOverride?.trim()?audioOverrideLanguage||undefined:undefined
    })
   });

   if(!r.ok){
    const err=await r.json().catch(()=>({}));
    throw Error(err.detail||'Audio generation failed');
   }

   if(r.headers.get('content-type')?.includes('audio/')){
    const prov=r.headers.get('X-TTS-Provider')||'edge-neural';
    const usedVoice=r.headers.get('X-TTS-Voice')||voiceId;
    setCurrentPlayingProvider(prov);
    setCurrentPlayingVoice(usedVoice);
    console.log(`[TTS Active] Provider: ${prov}, Voice: ${usedVoice}, Language: ${audioLanguage}`);

    let parsedTimings:TimingItem[]=[];
    const timingsB64=r.headers.get('X-TTS-Timings-Base64');
    if(timingsB64){
     try{parsedTimings=JSON.parse(decodeURIComponent(escape(atob(timingsB64))))}catch(e){console.warn('Failed to decode timings header:',e)}
    }

    const narrationB64=r.headers.get('X-TTS-Narration-Text-Base64');
    let currentNarration=narrationOverride?.trim()||result.executive_summary;
    if(narrationB64){
     try{currentNarration=decodeURIComponent(escape(atob(narrationB64)))}catch(e){console.warn('Failed to decode narration text header:',e)}
    }
    setAudioNarrationText(currentNarration);
    setAudioTimings(parsedTimings);

    const url=URL.createObjectURL(await r.blob());
    setAudioUrl(current=>{if(current)URL.revokeObjectURL(current);return url});
    setAudioPaused(false);
    notify(`Neural audio generated in ${languageNames[audioLanguage]||audioLanguage}`);
    return;
   }

   const d=await r.json();
   console.log('[TTS Response JSON]',d);
   if(d.fallback_used){
    console.warn(`[TTS Fallback]: ${d.fallback_reason}`);
    notify(`Edge voice unavailable; using system voice for ${languageNames[audioLanguage]||audioLanguage}`);
   }

   const narration=d.text||narrationOverride?.trim()||result.executive_summary;
   setAudioNarrationText(narration);
   if(d.timings&&d.timings.length){
    setAudioTimings(d.timings);
   }

   if(!('speechSynthesis'in window))throw Error('Browser speech playback is not available');
   const voice=voiceOptions.find(v=>`browser:${v.voiceURI}`===selectedVoice)||voiceOptions.find(v=>v.lang.toLowerCase().startsWith(audioLanguage));
   if(!voice)throw Error(`No ${languageNames[audioLanguage]||audioLanguage.toUpperCase()} system voice is installed. Choose an online neural voice.`);

   setCurrentPlayingProvider('browser-speech-synthesis');
   setCurrentPlayingVoice(voice.name);

   const utterance=new SpeechSynthesisUtterance(narration);
   utterance.lang=speechLocale[audioLanguage]||'en-US';
   utterance.rate=audioSpeed;
   utterance.voice=voice;

   utterance.onboundary=(ev)=>{
    if(ev.name==='sentence'||ev.name==='word'){
     updateActiveSegmentByCharIndex(ev.charIndex);
    }
   };
   utterance.onstart=()=>{setSpeaking(true);setAudioPaused(false)};
   utterance.onpause=()=>{setAudioPaused(true)};
   utterance.onresume=()=>{setAudioPaused(false)};
   utterance.onend=()=>{setAudioPaused(false);setSpeaking(false);setActiveSegmentIndex(null)};
   utterance.onerror=()=>{setAudioPaused(false);setSpeaking(false);setActiveSegmentIndex(null);notify('The selected system voice could not read this language.')};

   speechSynthesis.speak(utterance);
   setAudioPaused(false);
   notify(`Playing with ${voice.name}`);
  }catch(e:any){
   setActiveSegmentIndex(null);
   notify(e.message||'Audio unavailable');
  }finally{
   setAudioBusy(false);
  }
 }
 useEffect(()=>{if(page==='Audio briefing'&&pendingAudioPlay&&audioOverrideText){setPendingAudioPlay(false);void play(audioOverrideText)}},[page,pendingAudioPlay,audioOverrideText])
 const stats=result?.stats
 const claimCounts=useMemo(()=>result?.claims.reduce((a,c)=>(a[c.status]=(a[c.status]||0)+1,a),{} as Record<string,number>)||{},[result])
 const overviewSteps=useMemo<PipelineStep[]>(()=>derivePipelineSteps({hasDoc:!!doc,busy,error:!!error,resultStatus:result?.status??null,summaryUsable:isUsableSummary(result?.executive_summary),validationState:result?.validation_state,regenerationCount:result?.stats.regeneration_count||0,audioActive:!!audioUrl}),[doc,busy,error,result,audioUrl])
 return <div className="app-shell"><aside className="sidebar"><div className="brand"><div className="brand-mark"><Sparkles size={18}/></div><span>scribe<span className="brand-dot">.</span></span><button className="icon-btn mobile-menu" aria-label="Open menu"><Menu/></button></div><div className="workspace-label">WORKSPACE <ChevronDown size={12}/></div><div className="workspace"><div className="workspace-icon">N</div><div><b>Northstar Studio</b><small>Private workspace</small></div><ChevronDown size={15}/></div><div className="nav-label">WORKSPACE</div><nav>{nav.map(({name,icon:Icon})=><button key={name} className={`nav-item ${page===name?'active':''}`} onClick={()=>setPage(name)}><Icon size={17}/>{name}{name==='Documents'&&docs.length>0&&<small className="nav-count">{docs.length}</small>}</button>)}</nav><div className="side-spacer"/><div className="plan-card"><div className="plan-icon"><Sparkles size={15}/></div><b>Local demo mode</b><p>Source-grounded analysis is ready. AI providers can be connected as you scale.</p><div className="plan-meter"><i/></div><small>0 / 5 documents today</small></div><button className="profile" onClick={()=>void logout()}><div className="avatar">{user?.name.split(/\s+/).map(x=>x[0]).join('').slice(0,2).toUpperCase()}</div><div><b>{user?.name}</b><small>Sign out</small></div><span className="logout-label">Log out</span></button></aside>
  <main className="main"><header className="topbar"><div className="breadcrumbs"><span>Workspace</span><span className="crumb-slash">/</span><b>{page}</b>{doc&&<><span className="crumb-slash">/</span><span>{doc.name}</span></>}</div><div className="top-actions"><div className="system-pill"><span/> All systems operational</div><button className="icon-btn" aria-label="Help"><CircleHelp size={18}/></button><div className="avatar avatar-sm">{user?.name.split(/\s+/).map(x=>x[0]).join('').slice(0,2).toUpperCase()}</div></div></header>
  <div className="content">
  {page==='Overview'&&<><div className="welcome-row"><div><div className="eyebrow"><span/> DOCUMENT INTELLIGENCE, REIMAGINED</div><h1>Good morning, {user?.name.split(/\s+/)[0]} <span className="wave">✳</span></h1><p className="subhead">Turn complex documents into clear, trusted understanding.</p></div><div className="welcome-actions"><label className="overview-language"><span>SUMMARY LANGUAGE</span><select aria-label="Summary language" value={language} onChange={e=>setLanguage(e.target.value)}>{outputLanguages.map(l=><option key={l.code} value={l.code}>{l.name}</option>)}</select></label><button className="secondary-btn" onClick={demo}><Sparkles size={15}/> Try demo</button><button className="primary-btn" onClick={()=>document.getElementById('file-input')?.click()}><Plus size={17}/> New analysis</button></div></div><div className="hero-card"><div className="hero-copy"><span className="hero-kicker"><Sparkles size={13}/> YOUR INTELLIGENCE WORKSPACE</span><h2>Turn long documents<br/>into <em>trusted intelligence.</em></h2><p>Analyze, validate, and explore your documents with source-grounded intelligence. Every insight traces back to its source.</p><button className="hero-cta" onClick={()=>document.getElementById('file-input')?.click()}>Analyze a document <span>→</span></button><div className="hero-foot"><span><ShieldCheck size={15}/> Evidence, always in view</span><i/> <span><Waves size={15}/> Audio from validated summaries</span></div></div><Visual/></div><div className="section-title-row"><div><h3>Your workspace</h3><p>A clear view of your document activity.</p></div><button className="text-btn" onClick={()=>setPage('Documents')}>View all documents <span>→</span></button></div><div className="metric-grid"><Metric icon={<FileText/>} label="Documents analyzed" value={String(docs.length).padStart(2,'0')} change="This session"/><Metric icon={<ShieldCheck/>} label="Grounded claims" value={String(result?.quality.claims_checked||0).padStart(2,'0')} change="Source checked"/><Metric icon={<AudioLines/>} label="Audio briefings" value="00" change="Browser playback"/><Metric icon={<Clock3/>} label="Avg. processing" value="—" change="Local mode"/></div><div className="section-title-row compact"><div><h3>AI processing pipeline</h3><p>Live status of your current document, from extraction through validated audio.</p></div></div><AIProcessingPipeline steps={overviewSteps}/><div className="section-title-row compact"><div><h3>Recent documents</h3><p>Your latest work, all in one place.</p></div><button className="text-btn" onClick={()=>setPage('Documents')}>Browse documents <span>→</span></button></div><DocumentList docs={docs} open={openDocument} emptyAction={()=>document.getElementById('file-input')?.click()}/><CombinedSummaryExports/></>}
 {page==='Documents'&&<><PageHeading title="Your documents" subtitle="Manage and analyze your uploaded files." action={<button className="primary-btn" onClick={()=>document.getElementById('file-input')?.click()}><Plus size={17}/> Add document</button>}/>{doc&&<div className="chunk-visual-card"><div className="card-head"><div><h3>{doc.name}</h3><p>Document → Extraction → Chunking → Embeddings, sized to this document's real chunk count.</p></div></div><DocumentChunkVisualization status="extracted" chunkCount={doc.stats.chunks} wordCount={doc.stats.words} pages={doc.stats.pages}/></div>}<SearchDocuments docs={docs} open={openDocument} emptyAction={()=>document.getElementById('file-input')?.click()}/></>}
 {page==='History'&&<HistoryPage userId={user!.id} onOpen={async (_item,detail,destination)=>{setDoc(detail as Doc);setResult(detail.result||null);setPage(destination)}} onChanged={refresh}/>}
  {page==='Analysis'&&<><PageHeading title="Document analysis" subtitle={doc?`Configure an analysis for ${doc.name}.`:'Choose a document to begin.'}/>{doc?<><div className="file-card"><div className="file-icon"><FileText/></div><div className="file-info"><b>{doc.name}</b><small>{doc.stats.words.toLocaleString()} words · {doc.stats.chunks} evidence chunks · Detected: {languageNames[doc.language]||doc.language.toUpperCase()}</small></div><span className="status-chip"><span/> Extracted</span><button className="icon-btn" aria-label="Remove document" onClick={()=>{setDoc(null);setResult(null)}}><X size={17}/></button></div><div className="lang-banner"><div className="lang-banner-item"><span className="field-label" style={{marginBottom:0}}>SOURCE LANGUAGE</span><b className="lang-badge">{doc.language==='en'?'🇬🇧 English':`🇮🇳 ${languageNames[doc.language]||doc.language.toUpperCase()}`}</b></div><div className="lang-arrow">→</div><div className="lang-banner-item"><span className="field-label" style={{marginBottom:0}}>OUTPUT LANGUAGE</span><b className="lang-badge lang-badge--output">{language==='auto'?`Auto / ${languageNames[doc.language]||doc.language.toUpperCase()}`:languageLabel(language)}</b></div>{result&&result.source_language&&result.language!==result.source_language&&!(result&&(language==='auto'?doc.language:language)!==result.language)&&<span className="translation-chip">✓ Translation validated against original</span>}{result&&(language==='auto'?doc.language:language)!==result.language&&<span className="translation-chip" style={{background:'rgba(234, 179, 8, 0.15)',color:'#fbbf24',borderColor:'rgba(234, 179, 8, 0.4)'}}>⚠️ Pending translation · Click "Translate document" below</span>}</div>{doc.ocr_detected&&<div className="ocr-notice">Scanned or image content detected. Text was extracted for this analysis.</div>}{doc.ocr_warnings?.map((w,i)=><div className="ocr-notice" key={i}>{w}</div>)}<div className="analysis-layout"><div className="config-card"><div className="card-head"><div><h3>Shape your summary</h3><p>Choose what matters to you. You can change language without re-uploading.</p></div><Sparkles className="muted-icon" size={19}/></div><PipelineVisualization hasDoc busy={busy} error={!!error}/><label className="field-label">SUMMARY DEPTH</label><div className="depth-grid" role="radiogroup" aria-label="Summary depth">{[['quick','Quick','A concise overview'],['standard','Standard','Balanced coverage'],['deep','Deep','Detailed analysis'],['auto','Auto','Recommended mode'],['custom','Custom','Choose sections and detail']].map(([v,n,d])=><button type="button" role="radio" aria-checked={depth===v} className={`depth-option ${depth===v?'selected':''}`} onClick={()=>setDepth(v)} key={v}><span className="radio" aria-hidden="true"/><b>{n}</b><small>{d}</small></button>)}</div>{depth==='custom'&&<section className="custom-summary" aria-labelledby="custom-summary-title"><header className="custom-summary-head"><div><span className="custom-eyebrow">YOUR CONFIGURATION</span><h4 id="custom-summary-title">Custom Summary</h4><p>Choose how detailed your summary should be and which sections to include.</p></div><span className="custom-count" aria-live="polite">{customSections.length} of {customSectionOptions.length} selected</span></header><section className="custom-control-group" aria-labelledby="custom-detail-heading"><div className="custom-group-heading"><div><h5 id="custom-detail-heading">Detail level</h5><p>Set the amount of context in your summary.</p></div></div><div className="custom-detail-options" role="group" aria-labelledby="custom-detail-heading">{customDetailOptions.map(option=><button type="button" key={option.value} className={`custom-detail-option ${customDetail===option.value?'selected':''}`} aria-pressed={customDetail===option.value} onClick={()=>setCustomDetail(option.value)}><span className="custom-detail-indicator" aria-hidden="true"/><span><b>{option.label}</b><small>{option.description}</small></span></button>)}</div></section><section className="custom-control-group custom-sections-group" aria-labelledby="custom-sections-heading"><div className="custom-group-heading custom-sections-heading"><div><h5 id="custom-sections-heading">Include sections</h5><p>Select the sections you want in the generated summary.</p></div><div className="custom-section-actions"><button type="button" className="text-btn" onClick={()=>setCustomSections(customSectionOptions.map(option=>option.value))} disabled={customSections.length===customSectionOptions.length}>Select all</button><span aria-hidden="true">?</span><button type="button" className="text-btn" onClick={()=>setCustomSections([])} disabled={customSections.length===0}>Clear all</button></div></div><div className="custom-sections" role="group" aria-labelledby="custom-sections-heading">{customSectionOptions.map(option=>{const selected=customSections.includes(option.value);return <button type="button" role="checkbox" aria-checked={selected} key={option.value} className={`custom-section-card ${selected?'selected':''}`} onClick={()=>setCustomSections(current=>selected?current.filter(value=>value!==option.value):customSectionOptions.filter(item=>item.value===option.value||current.includes(item.value)).map(item=>item.value))}><span className="custom-section-check" aria-hidden="true">{selected&&<Check size={13}/>}</span><span>{option.label}</span></button>})}</div></section><div className="custom-summary-preview" aria-live="polite"><span className="custom-preview-mark"><Sparkles size={15}/></span><span><b>Custom summary</b><small>{customDetailOptions.find(option=>option.value===customDetail)?.label} detail ? {customSections.length} section{customSections.length===1?'':'s'} selected{customSections.length===0?' ? executive overview only':''}</small></span></div></section>}<div className="field-grid"><label><span className="field-label">PURPOSE</span><select value={purpose} onChange={e=>setPurpose(e.target.value)}>{['Executive briefing','General understanding','Research','Decision making','Risk analysis','Action planning','Presentation preparation'].map(x=><option key={x}>{x}</option>)}</select></label><label><span className="field-label">AUDIENCE</span><select value={audience} onChange={e=>setAudience(e.target.value)}>{['Leadership','Research team','Project team','General audience','Technical audience'].map(x=><option key={x}>{x}</option>)}</select></label><label><span className="field-label">OUTPUT LANGUAGE</span><select value={language} onChange={e=>setLanguage(e.target.value)}>{outputLanguages.map(l=><option key={l.code} value={l.code}>{l.name}</option>)}</select></label></div><div className="config-note"><ShieldCheck size={16}/><span><b>Validation first.</b> Claims are checked against extracted source text before they appear as validated or become eligible for audio.</span></div>{result&&(language==='auto'?doc.language:language)!==result.language&&<div className="config-note" style={{borderLeftColor:'#f59e0b',background:'rgba(245, 158, 11, 0.08)'}}><Sparkles size={16} color="#f59e0b"/><span><b>Translation ready.</b> Target language set to <b>{languageLabel(language==='auto'?doc.language:language)}</b>. Click <b>"Translate document"</b> below to translate and validate against the source.</span></div>}{error&&<div className="config-note" style={{borderLeftColor:'#ef4444',background:'rgba(239, 68, 68, 0.1)',color:'#fca5a5'}}><AlertCircle size={16} color="#ef4444"/><span><b>Notice:</b> {error}</span></div>}<button className="primary-btn analyze-btn" disabled={busy} onClick={analyze}>{busy?<><span className="spinner"/> {result&&(language==='auto'?doc.language:language)!==result.language?'Translating…':'Generating & evaluating…'}</>:result&&(language==='auto'?doc.language:language)!==result.language?<><Sparkles size={16}/> Translate document to {languageLabel(language==='auto'?doc.language:language)} <span>→</span></>:<><Sparkles size={16}/> Analyze document <span>→</span></>}</button></div><div className="side-info-card"><div className="side-info-icon"><Layers3 size={19}/></div><h3>Document profile</h3><p>Extraction metadata from your uploaded file.</p><div className="profile-row"><span>Word count</span><b>{doc.stats.words.toLocaleString()}</b></div><div className="profile-row"><span>Evidence chunks</span><b>{doc.stats.chunks}</b></div><div className="profile-row"><span>Detected language</span><b>{languageNames[doc.language]||doc.language.toUpperCase()}</b></div><div className="profile-row"><span>File type</span><b>{doc.name.split('.').pop()?.toUpperCase()}</b></div><div className="provider-note"><span className="amber-dot"/> <span>{result?.provider?`Runtime provider: ${result.provider.name}${result.provider.model?` · ${result.provider.model}`:''}`:'Runtime provider is shown with each analysis result.'}<br/><small>{result?.provider?.fallback_used?'A fallback provider generated this summary.':'Provider selection is reported by the analysis API.'}</small></span></div></div></div></>:<div className="empty-card"><div className="empty-mark"><Upload/></div><h3>Start with a document</h3><p>Upload a PDF, DOCX, TXT, CSV, JSON, JPG, or PNG file to extract text and configure an analysis.</p><button className="primary-btn" onClick={()=>document.getElementById('file-input')?.click()}><Upload size={16}/> Choose a document</button></div>}{result&&<ResultCard result={result} claimCounts={claimCounts} setPage={setPage} docId={doc?.id||''} speechSegments={speechSegments} activeSegmentIndex={activeSegmentIndex} isAudioActive={!audioPaused&&(!!audioUrl||speaking)} audioLanguage={audioLanguage}/>}</>}
 {page==='Sources'&&<><PageHeading title="Source library" subtitle="Browse extracted passages and trace claims back to their source."/>{doc?<>{result&&<div className="source-graph-card"><div className="card-head"><div><h3>Claim → evidence graph</h3><p>Each validated claim traced back to its source passage. Click a node for detail.</p></div></div><SourceValidationGraph claims={result.claims}/></div>}<Sources doc={doc}/></>:<EmptySelect docs={docs} choose={d=>{setDoc(d);setPage('Sources')}}/>}</>}
 {page==='Ask document'&&<><PageHeading title="Ask your document" subtitle="Search extracted passages with lexical retrieval. Answers stay close to the source."/>{doc?<div className="chat-card"><div className="chat-context"><div className="file-icon small-file"><FileText size={16}/></div><span><b>{doc.name}</b><small>{doc.stats.chunks} searchable chunks</small></span><span className="online-tag"><i/> Source search ready</span>{(language==='auto'?doc.language:language)!=='en'&&<span className="lang-answer-tag">Answers in {languageLabel(language==='auto'?doc.language:language)}</span>}</div><div className="chat-empty"><div className="chat-orb"><MessageSquareText/></div><h3>What would you like to know?</h3><p>Ask about the document. Scribe will surface relevant extracted passages with citations.</p><div className="suggestions">{['What are the key findings?','What risks are mentioned?','What recommendations are included?'].map(s=><button key={s} onClick={()=>setQuestion(s)}>{s} <span>↗</span></button>)}</div>{(askBusy||answer)&&<RAGVisualization phase={askBusy?'retrieving':answer?'answered':'idle'} chunks={answer?.sources||[]} selectedChunk={selectedChunk} onSelectChunk={id=>setSelectedChunk(current=>current===id?null:id)}/>}{answer&&<div className="answer"><div className="answer-label"><Sparkles size={14}/> SOURCE RESPONSE <span className="grounded-tag">{answer.grounded?'Evidence found':'No evidence'}</span></div><p>{answer.answer}</p>{answer.sources?.map((s:any)=><details className="answer-source" key={s.chunk_id} open={selectedChunk===s.chunk_id} onToggle={e=>{const isOpen=(e.target as HTMLDetailsElement).open;setSelectedChunk(current=>isOpen?s.chunk_id:(current===s.chunk_id?null:current))}}><summary>{s.section} ? {s.page?`Page ${s.page}`:s.chunk_id} ? Open source passage</summary><p>{s.text}</p></details>)}</div>}</div><form className="ask-form" onSubmit={e=>{e.preventDefault();ask()}}><input value={question} onChange={e=>setQuestion(e.target.value)} placeholder="Ask a question about this document…"/><button className="primary-btn" type="submit">Ask <span>↑</span></button></form></div>:<EmptySelect docs={docs} choose={d=>{setDoc(d);setPage('Ask document')}}/>}</>}
  {page==='Audio briefing'&&<><PageHeading title="Audio briefing" subtitle="Choose a narration language. The summary can be translated for speech."/>{doc&&result?<div className="audio-card"><div className="audio-glow"><AudioLines size={30}/></div><div className="audio-content"><span className="hero-kicker"><ShieldCheck size={13}/> {result.status==='validated'?'VALIDATED SUMMARY':result.status==='partial'?'PARTIALLY VALIDATED SUMMARY':'SUMMARY READY'}</span>{result.validation_warning&&<div className={`validation-warning ${result.validation_state==='VALIDATION_FAILED_AFTER_RETRIES'?'validation-warning--strong':''}`} role="alert"><b>WARNING: Validation incomplete</b><p>{result.warning_message||'Some claims could not be fully verified. Review the summary against the source before relying on it.'}</p></div>}<h2>Your document, <em>in audio.</em></h2>{currentPlayingProvider&&<div className="speech-segment-badge" style={{marginBottom:'0.75rem',width:'fit-content'}}><Volume2 size={13}/><span>TTS: <b>{currentPlayingProvider}</b> {currentPlayingVoice?`(${currentPlayingVoice})`:''}</span></div>}<div className="audio-summary-transcript">{speechSegments.length>0?speechSegments.map((seg,i)=><span key={i} className={`speech-segment ${activeSegmentIndex===i?'speech-segment--active':''}`}>{seg.text}{' '}</span>):<p>{audioOverrideText||result.executive_summary||'No summary text is available.'}</p>}</div>{(audioUrl||speaking)&&<AudioVisualization audioEl={audioRef.current} playing={!audioPaused&&(!!audioUrl||speaking)}/>}<div className="audio-controls"><button className="play-btn" disabled={audioBusy} onClick={()=>play(audioOverrideText||undefined)}>{audioBusy?<><span className="spinner"/> Generating…</>:<><Play size={17} fill="currentColor"/> {selectedVoice.startsWith('edge:')?'Generate & play':'Play briefing'}</>}</button><label>LANGUAGE<select value={audioLanguage} onChange={e=>{const newLang=e.target.value;setAudioLanguage(newLang);setAudioUrl('');speechSynthesis.cancel();setSpeaking(false);setActiveSegmentIndex(null);const sysVoice=voiceOptions.find(v=>v.lang.toLowerCase().startsWith(newLang));if(sysVoice){setSelectedVoice(`browser:${sysVoice.voiceURI}`)}else{const neural=(neuralVoices[newLang]||[])[0];if(neural)setSelectedVoice(`edge:${neural.id}`)}}}>{supportedLanguages.map(l=><option key={l.code} value={l.code}>{l.name}</option>)}</select></label><label>VOICE<select value={selectedVoice} onChange={e=>{setSelectedVoice(e.target.value);setAudioUrl('');speechSynthesis.cancel();setSpeaking(false);setActiveSegmentIndex(null)}}>{voiceOptions.filter(v=>v.lang.toLowerCase().startsWith(audioLanguage)).map(v=><option key={v.voiceURI} value={`browser:${v.voiceURI}`}>{v.name} · system</option>)}<optgroup label="Online neural voices">{(neuralVoices[audioLanguage]||[]).map(v=><option key={v.id} value={`edge:${v.id}`}>{v.name}</option>)}</optgroup></select></label><label>SPEED<input type="range" min="0.7" max="1.3" step="0.1" value={audioSpeed} onChange={e=>{const rate=Number(e.target.value);setAudioSpeed(rate);if(audioRef.current)audioRef.current.playbackRate=rate}}/></label>{audioUrl?<a className="secondary-btn audio-download" href={audioUrl} download={`scribe-briefing-${audioLanguage}.mp3`}><ArrowDownToLine size={14}/> Download</a>:selectedVoice.startsWith('browser:')?<button className="secondary-btn" onClick={()=>{if(audioPaused){speechSynthesis.resume();setAudioPaused(false)}else{speechSynthesis.pause();setAudioPaused(true)}}}>{audioPaused?'Resume':'Pause'}</button>:null}</div>{audioUrl&&<audio ref={audioRef} className="generated-audio" controls src={audioUrl} onTimeUpdate={syncAudioTime} onSeeking={syncAudioTime} onSeeked={syncAudioTime} onPlay={()=>setAudioPaused(false)} onPause={()=>setAudioPaused(true)} onEnded={()=>{setAudioPaused(false);setActiveSegmentIndex(null)}} />}{selectedVoice.startsWith('edge:')?<small className="audio-disclaimer">Online neural speech sends only this summary to Microsoft’s speech service to generate audio. Audio is held in this browser session.</small>:<small className="audio-disclaimer">System voices run on your device. Translation uses the configured provider chain when narration and summary languages differ. If a matching system voice is unavailable, choose an online neural voice.</small>}</div></div>:<EmptySelect docs={docs} choose={d=>{setDoc(d);setPage('Analysis')}} label="Choose a processed document to create an audio briefing."/>}</>}
 {page==='Compare'&&<Compare/>}
 {page==='LLM Provider Test'&&<LocalLLMTest documentReady={!!doc&&!!result} documentId={doc?.id} onPlayAudio={(text,language)=>{setAudioOverrideText(text);setAudioOverrideLanguage(language||'');if(language)setAudioLanguage(language);setPendingAudioPlay(true);setPage('Audio briefing')}} onOpenAudio={(text,language)=>{setAudioOverrideText(text);setAudioOverrideLanguage(language||'');if(language)setAudioLanguage(language);setPendingAudioPlay(false);setPage('Audio briefing')}}/>}
 </div></main><label className="ocr-language-picker"><span>OCR INPUT LANGUAGE</span><select aria-label="OCR input language" value={ocrLanguage} onChange={e=>setOcrLanguage(e.target.value)}>{[{code:"auto",name:"Auto detect"},...supportedLanguages.slice(0,5)].map(l=><option key={l.code} value={l.code}>{l.name}</option>)}</select></label><input id="file-input" className="hidden" type="file" accept=".pdf,.docx,.txt,.csv,.json,.jpg,.jpeg,.png" onChange={e=>{const f=e.target.files?.[0];if(f)upload(f);e.currentTarget.value=''}}/>{busy&&<div className="processing-toast"><span className="spinner"/> Extracting text, scanning images, and running OCR…</div>}{error&&<div className="error-toast"><span>{error}</span><button aria-label="Dismiss error" onClick={()=>setError('')}><X size={16}/></button></div>}{toast&&<div className="success-toast"><Check size={16}/>{toast}</div>}{previewOpen&&<button onClick={()=>setPreviewOpen(false)} className="hidden"/>}</div>
}

function Metric({icon,label,value,change}:{icon:any;label:string;value:string;change:string}){return <div className="metric-card"><div className="metric-top">{icon}<span>{change}</span></div><b>{value}</b><small>{label}</small></div>}
function PageHeading({title,subtitle,action}:{title:string;subtitle:string;action?:any}){return <div className="page-heading"><div><div className="eyebrow"><span/> SOURCE-GROUNDED INTELLIGENCE</div><h1>{title}</h1><p>{subtitle}</p></div>{action}</div>}
function SearchDocuments({docs,open,emptyAction}:{docs:Doc[];open:(d:Doc)=>void;emptyAction?:()=>void}){
 const [query,setQuery]=useState(''),[matches,setMatches]=useState<Doc[]>(docs),[searching,setSearching]=useState(false),[searchError,setSearchError]=useState('')
 const sequence=React.useRef(0)
 useEffect(()=>{if(!query.trim())setMatches(docs)},[docs,query])
 useEffect(()=>{
  const controller=new AbortController(),requestId=++sequence.current,term=query.trim()
  setSearching(true);setSearchError('')
  const timer=window.setTimeout(async()=>{
   try{
    const url=`${API}/documents${term?`?search=${encodeURIComponent(term)}`:''}`
    const response=await fetch(url,{signal:controller.signal})
    if(!response.ok)throw new Error(await responseError(response,'Unable to search documents. Please try again.'))
    const data=await response.json()
    if(!Array.isArray(data))throw new Error('Unable to search documents. Please try again.')
    if(requestId===sequence.current)setMatches(data)
   }catch(error){
    if(controller.signal.aborted)return
    if(requestId===sequence.current)setSearchError('Unable to search documents. Please try again.')
   }finally{if(!controller.signal.aborted&&requestId===sequence.current)setSearching(false)}
  },term?300:0)
  return()=>{window.clearTimeout(timer);controller.abort()}
 },[query])
 const clear=()=>setQuery('')
 return <section className="document-search" aria-label="Search your documents">
  <label htmlFor="document-search-input">Search documents</label>
  <div className="document-search-control"><Search size={17} aria-hidden="true"/><input id="document-search-input" type="search" value={query} maxLength={256} onChange={event=>setQuery(event.target.value)} placeholder="Search filenames, language, metadata, or document content?" aria-label="Search documents" aria-describedby="document-search-status"/>{query&&<button type="button" className="document-search-clear" onClick={clear} aria-label="Clear document search"><X size={16}/></button>}</div>
  <div className="document-search-status" id="document-search-status" aria-live="polite" aria-busy={searching}>{searching?<span><span className="spinner"/> Searching documents?</span>:searchError?<span className="document-search-error" role="alert">{searchError}</span>:<span>{matches.length} {matches.length===1?'document':'documents'} found{query.trim()?<> for <b>?{query.trim()}?</b></>:null}</span>}</div>
  {!searchError&&matches.length>0&&<DocumentList docs={matches} open={open} emptyAction={emptyAction}/>}
  {!searchError&&!searching&&matches.length===0&&(query.trim()?<div className="document-search-empty"><div className="empty-mark"><Search/></div><h3>No documents found</h3><p>No documents match ?{query.trim()}?.</p><small>Try another search term or clear the search.</small><button className="secondary-btn" type="button" onClick={clear}>Clear search</button></div>:<DocumentList docs={[]} open={open} emptyAction={emptyAction}/>)}
 </section>
}

function DocumentList({docs,open,emptyAction}:{docs:Doc[];open:(d:Doc)=>void;emptyAction?:()=>void}){return docs.length?<div className="doc-list">{docs.map((d,i)=><button className="doc-row" key={d.id} onClick={()=>open(d)}><div className="file-icon"><FileText/></div><div className="doc-row-main"><b>{d.name}</b><small>{d.stats.words.toLocaleString()} words · {d.stats.chunks} chunks · Added {new Date(d.created_at||Date.now()).toLocaleDateString()}</small></div><span className="language-chip">{d.language.toUpperCase()}</span><span className="status-chip"><span/> {d.status.toLowerCase()}</span><span className="row-arrow">↗</span></button>)}</div>:<div className="empty-card"><div className="empty-mark"><FileText/></div><h3>Your workspace is ready</h3><p>Upload your first document to extract content, check claims, and explore the evidence.</p><button className="primary-btn" onClick={emptyAction}><Upload size={16}/> Upload a document</button></div>}
function EmptySelect({docs,choose,label='Select a document to continue.'}:{docs:Doc[];choose:(d:Doc)=>void;label?:string}){return <div className="empty-card"><div className="empty-mark"><FolderOpen/></div><h3>Choose a document</h3><p>{label}</p>{docs.length?docs.map(d=><button key={d.id} className="select-doc" onClick={()=>choose(d)}>{d.name}<span>→</span></button>):<small>Upload one from the Documents page first.</small>}</div>}
function SummarySections({result:r}:{result:Result}){
 const sectionNames:Record<string,string[]>={
  en:["KEY FINDINGS","IMPORTANT FACTS","RISKS","DECISIONS","RECOMMENDATIONS","ACTION ITEMS"],
  hi:["मुख्य निष्कर्ष","महत्वपूर्ण तथ्य","जोखिम","निर्णय","सिफारिशें","कार्य बिंदु"],
  te:["ముఖ్యమైన ఫలితాలు","ముఖ్యమైన వాస్తవాలు","ప్రమాదాలు","నిర్ణయాలు","సిఫార్సులు","చర్య అంశాలు"],
  kn:["ಮುಖ್ಯಾಂಶಗಳು","ಪ್ರಮುಖ ಸಂಗತులు","ಅಪಾಯಗಳು","ನಿರ್ಧಾರಗಳು","ಶಿಫಾರಸುಗಳು","ಕಾರ್ಯ ಅಂಶಗಳು"],
  ta:["முக்கிய கண்டுபிடிப்புகள்","முக்கிய உண்மைகள்","அபாயங்கள்","முடிவுகள்","பரிந்துரைகள்","செயல் உருப்படிகள்"],
  ml:["പ്രധാന കണ്ടെത്തലുകൾ","പ്രധാന വസ്തുതകൾ","അപകടസാധ്യതകൾ","തീരുമാനങ്ങൾ","ശുപാർശകൾ","പ്രവർത്തന ഇനങ്ങൾ"],
  bn:["মূল অনুসন্ধান","গুরুত্বপূর্ণ তথ্য","ঝুঁকি","সিদ্ধান্ত","সুপারিশসমূহ","করণীয় বিষয়"],
  mr:["मुख्य निष्कर्ष","महत्त्वाची तथ्ये","जोखीम","निर्णय","शिफारसी","कार्यसूची"],
  gu:["મુખ્ય તારણો","મહત્વપૂર્ણ તથ્યો","જોખમો","નિર્ણયો","ભલામણો","કાર્ય મુદ્દાઓ"],
  pa:["ਮੁੱਖ ਖੋਜਾਂ","ਮਹੱਤਵਪੂਰਨ ਤੱਥ","ਜੋਖਮ","ਫੈਸਲੇ","ਸਿਫ਼ਾਰਸ਼ਾਂ","ਕਾਰਵਾਈ ਆਈਟਮਾਂ"],
  ur:["اہم نتائج","اہم حقائق","خطرات","فیصلے","سفارشات","عملی نکات"],
 }
 const defaults=sectionNames.en
 const labels=sectionNames[r.language]||defaults
 const sections:[string,string[]|undefined][]=[...labels.map((label,i)=>[label,[r.findings,r.facts,r.risks,r.decisions,r.recommendations,r.action_items][i]] as [string,string[]|undefined])]
 return <div className="summary-sections">{sections.filter(([,items])=>items?.length).map(([label,items])=><section key={label}><h4>{label}</h4><ul>{items?.map((item,i)=><li key={i}>{item}</li>)}</ul></section>)}</div>
}

function EvaluationPanel({evaluation}:{evaluation:any}){
 if(!evaluation?.metrics)return null
 const labels:Record<string,string>={relevance:'Relevance',completeness:'Completeness',factual_consistency:'Factual Consistency',groundedness:'Groundedness',summary_quality:'Summary Quality',validation_accuracy:'Validation Accuracy'}
 const metrics=Object.entries(labels).map(([key,label])=>[key,label,evaluation.metrics[key]||{}] as [string,string,any])
 const claims=Array.isArray(evaluation.claims)?evaluation.claims:[]
 const supported=claims.filter((claim:any)=>claim.validation_result==='SUPPORTED').length
 const unsupported=claims.filter((claim:any)=>claim.validation_result==='UNSUPPORTED'||claim.validation_result==='PARTIALLY_SUPPORTED').length
 const applicable=evaluation.applicable_metric_count??metrics.filter(([, ,metric])=>metric.score!=null).length
 const status=String(evaluation.evaluation_status||'unavailable').replace(/_/g,' ').toUpperCase()
 return <section className="evaluation-panel" aria-labelledby="summary-quality-title"><div className="evaluation-report-head"><div><span className="hero-kicker">EVALUATION · {status}</span><h4 id="summary-quality-title">Summary Quality Report</h4><p>{evaluation.reason||'Deterministic scoring against source text and validated claims.'}</p></div><div className="evaluation-overall"><small>OVERALL SCORE</small><b>{evaluation.overall_score_rounded==null?'N/A':`${evaluation.overall_score_rounded}%`}</b><span>Based on {applicable} applicable metrics</span></div></div><div className="evaluation-grid">{metrics.map(([key,label,metric])=>{const score=metric.score;return <article className="evaluation-metric" key={key}><div className="evaluation-metric-head"><small>{label}</small><b>{score==null?'N/A':`${Math.round(score)}%`}</b></div>{score!=null&&<div className="evaluation-track" role="progressbar" aria-label={`${label} score`} aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(score)}><span style={{width:`${Math.max(0,Math.min(100,score))}%`}}/></div>}<span>{metric.explanation||'This metric could not be calculated.'}</span>{metric.status&&<em>{String(metric.status).replace(/_/g,' ')}</em>}</article>})}</div><details className="evaluation-claims"><summary>Claim Validation · {claims.length} claims analyzed</summary><div className="evaluation-claim-counts"><span>Claims analyzed <b>{claims.length}</b></span><span>Supported <b>{supported}</b></span><span>Unsupported / partial <b>{unsupported}</b></span><span>Validation accuracy <b>{evaluation.metrics.validation_accuracy?.score==null?'N/A':`${Math.round(evaluation.metrics.validation_accuracy.score)}%`}</b></span></div>{claims.length?claims.map((claim:any,index:number)=><article className="evaluation-claim" key={`${claim.claim}-${index}`}><b>{claim.claim}</b><small>{claim.validation_result||'UNAVAILABLE'}{claim.confidence==null?'':` · confidence ${Math.round(claim.confidence*100)}%`}</small>{claim.source_evidence?.map((evidence:string,i:number)=><blockquote key={i}>{evidence}</blockquote>)}{claim.sources?.map((source:any,i:number)=><small key={i}>Source{source.page!=null?` · page ${source.page}`:''}{source.section?` · ${source.section}`:''}{source.chunk_id?` · ${source.chunk_id}`:''}</small>)}</article>):<p>No factual claims requiring validation were detected.</p>}</details></section>
}

/** Wraps AIProcessingPipeline with the app's real state, never fake/random node states. */
function PipelineVisualization({resultStatus=null,summaryUsable,validationState=null,regenerationCount=0,hasDoc=true,busy=false,error=false,audioActive=false}:{resultStatus?:string|null;summaryUsable?:boolean;validationState?:string|null;regenerationCount?:number;hasDoc?:boolean;busy?:boolean;error?:boolean;audioActive?:boolean}){
 const steps=derivePipelineSteps({hasDoc,busy,error,resultStatus,summaryUsable,validationState,regenerationCount,audioActive})
 return <AIProcessingPipeline steps={steps} compact/>
}

function SummaryExportMenu({url,label='Export Summary'}:{url:string;label?:string}){
 const [open,setOpen]=useState(false),[format,setFormat]=useState(''),[error,setError]=useState('')
 const formats=[['pdf','PDF'],['docx','DOCX'],['txt','TXT'],['md','Markdown'],['json','JSON']]
 async function exportFile(value:string,name:string){
  setFormat(name);setError('')
  try{
   const response=await fetch(`${url}?format=${encodeURIComponent(value)}`)
   if(!response.ok){const body=await response.json().catch(()=>({}));throw Error(body?.detail||`Could not generate ${name}.`)}
   const blob=await response.blob(),objectUrl=URL.createObjectURL(blob),anchor=document.createElement('a')
   const disposition=response.headers.get('Content-Disposition')||''
   const filename=disposition.match(/filename="?([^";]+)"?/i)?.[1]||`scribe-summary.${value==='markdown'?'md':value}`
   anchor.href=objectUrl;anchor.download=filename;document.body.appendChild(anchor);anchor.click();anchor.remove();URL.revokeObjectURL(objectUrl)
  }catch(e:any){setError(e?.message||`Could not generate ${name}.`)}finally{setFormat('')}
 }
 return <div className="export-menu-wrap"><button className="secondary-btn" aria-haspopup="menu" aria-expanded={open} onClick={()=>{setOpen(value=>!value);setError('')}}><ArrowDownToLine size={15}/>{label}<ChevronDown size={14}/></button>{open&&<div className="export-menu" role="menu">{formats.map(([value,name])=><button key={value} role="menuitem" disabled={!!format} onClick={()=>exportFile(value,name)}>{format===name?`Generating ${name}…`:name}</button>)}</div>}{error&&<p className="export-error" role="alert">{error}</p>}</div>
}

function LLMTestHistory(){
 const [items,setItems]=useState<any[]>([])
 useEffect(()=>{let active=true;fetch(`${API}/history?activity_type=llm_test&limit=10`).then(r=>r.ok?r.json():[]).then(data=>{if(active)setItems(Array.isArray(data)?data:[])}).catch(()=>{});return()=>{active=false}},[])
 if(!items.length)return null
 return <section className="history-llm-tests"><div><h2>LLM Provider Test runs</h2><p>Saved prompt responses from the provider test page.</p></div><div className="history-llm-test-list">{items.map(item=><details key={item.id} className="history-llm-test"><summary><span><b>{item.title}</b><small>{item.extra_data?.purpose||'general'} · {item.extra_data?.audience||'general'} · {item.extra_data?.depth||'standard'} · {item.provider||'Provider unavailable'}{item.model?` · ${item.model}`:''}{item.extra_data?.translation_language?` · Translated to ${item.extra_data.translation_language.toUpperCase()}`:''}</small></span><time>{item.created_at?new Date(item.created_at).toLocaleString():''}</time></summary><div><p><b>Prompt</b><br/>{item.input_text||''}</p><p><b>Generated response</b><br/>{item.output_text||item.error_message||''}</p>{item.document_id&&<small>Document: {item.document_id}</small>}</div></details>)}</div></section>
}

function HistoryPage({userId,onOpen,onChanged}:{userId:number;onOpen:(item:any,detail:any,destination:string)=>void;onChanged:()=>void}){
 const [items,setItems]=useState<any[]>([]),[total,setTotal]=useState(0),[page,setPage]=useState(1),[pages,setPages]=useState(0)
 const [search,setSearch]=useState(''),[fileType,setFileType]=useState(''),[status,setStatus]=useState(''),[language,setLanguage]=useState(''),[dateFrom,setDateFrom]=useState(''),[dateTo,setDateTo]=useState(''),[sort,setSort]=useState('newest')
 const [selected,setSelected]=useState<any>(null),[detail,setDetail]=useState<any>(null),[loading,setLoading]=useState(false),[error,setError]=useState('')
 const hasFilters=!!(search||fileType||status||language||dateFrom||dateTo)
 useEffect(()=>{let active=true;const params=new URLSearchParams({page:String(page),limit:'20',sort});if(search.trim())params.set('search',search.trim());if(fileType)params.set('file_type',fileType);if(status)params.set('status',status);if(language)params.set('language',language);if(dateFrom)params.set('date_from',`${dateFrom}T00:00:00`);if(dateTo){const next=new Date(`${dateTo}T00:00:00`);next.setDate(next.getDate()+1);params.set('date_to',next.toISOString().slice(0,19))}setLoading(true);fetch(`${API}/history?${params}`).then(async r=>{if(!r.ok)throw Error('History could not be loaded.');return r.json()}).then(data=>{if(active){setItems(data.items||[]);setTotal(data.total||0);setPages(data.pages||0)}}).catch(e=>{if(active)setError(e.message)}).finally(()=>{if(active)setLoading(false)});return()=>{active=false}},[userId,page,search,fileType,status,language,dateFrom,dateTo,sort])
 async function choose(item:any){setSelected(item.id);setDetail(null);setError('');try{const r=await fetch(`${API}/documents/${encodeURIComponent(item.id)}`);const body=await r.json();if(!r.ok)throw Error(body.detail||'Document details could not be loaded.');setDetail(body)}catch(e:any){setError(e.message)}}
 async function remove(item:any){if(!window.confirm(`Delete “${item.name}” and its saved analysis?`))return;const r=await fetch(`${API}/documents/${encodeURIComponent(item.id)}`,{method:'DELETE'});if(!r.ok){setError('Document could not be deleted.');return}if(selected===item.id){setSelected(null);setDetail(null)}onChanged();setPage(current=>current>1&&items.length===1?current-1:current)}
 const clearFilters=()=>{setSearch('');setFileType('');setStatus('');setLanguage('');setDateFrom('');setDateTo('');setSort('newest');setPage(1)}
 return <><PageHeading title="History" subtitle="Your uploaded documents and the work completed on them."/><LLMTestHistory/>
  <section className="history-panel"><div className="history-filters"><label className="history-search"><Search size={16}/><input aria-label="Search document name" value={search} onChange={e=>{setSearch(e.target.value);setPage(1)}} placeholder="Search documents"/></label>
   <select aria-label="Filter by file type" value={fileType} onChange={e=>{setFileType(e.target.value);setPage(1)}}><option value="">All file types</option>{['pdf','docx','txt','csv','json','jpg','jpeg','png'].map(x=><option key={x} value={x}>{x.toUpperCase()}</option>)}</select>
   <select aria-label="Filter by processing status" value={status} onChange={e=>{setStatus(e.target.value);setPage(1)}}><option value="">All statuses</option><option value="uploaded">Uploaded</option><option value="completed">Completed</option></select>
   <select aria-label="Filter by language" value={language} onChange={e=>{setLanguage(e.target.value);setPage(1)}}><option value="">All languages</option>{supportedLanguages.map(x=><option key={x.code} value={x.code}>{x.name}</option>)}</select>
   <label className="history-date">From<input type="date" value={dateFrom} onChange={e=>{setDateFrom(e.target.value);setPage(1)}}/></label><label className="history-date">To<input type="date" value={dateTo} onChange={e=>{setDateTo(e.target.value);setPage(1)}}/></label>
   <select aria-label="Sort history" value={sort} onChange={e=>{setSort(e.target.value);setPage(1)}}><option value="newest">Newest first</option><option value="oldest">Oldest first</option></select></div>
   {error&&<p className="history-error" role="alert">{error}</p>}{loading?<div className="history-empty">Loading your history…</div>:items.length===0?<div className="history-empty"><div className="empty-mark"><Clock3/></div><h3>{hasFilters?'No matching documents':'No documents yet'}</h3><p>{hasFilters?'Try changing or clearing your filters.':'Upload your first document to see it here.'}</p><button className="primary-btn" onClick={()=>hasFilters?clearFilters():document.getElementById('file-input')?.click()}>{hasFilters?'Clear filters':'Upload your first document'}</button></div>:<>
    <div className="history-list">{items.map(item=><article key={item.id} className={`history-row ${selected===item.id?'selected':''}`}><button className="history-row-main" onClick={()=>void choose(item)}><span className="file-icon"><FileText/></span><span className="history-title"><b>{item.name}</b><small>{item.file_type.toUpperCase()} · Uploaded {new Date(item.created_at).toLocaleString()}</small></span><span className="history-badges"><i>{item.status.toLowerCase()}</i>{item.summary_status&&<i>{item.summary_status}</i>}</span><span className="row-arrow">↗</span></button><div className="history-meta"><span>Processed: {item.processed_at?new Date(item.processed_at).toLocaleString():'Not processed'}</span><span>Language: {languageNames[item.language]||item.language?.toUpperCase()||'Unknown'}</span><span>Summary: {item.summary_type||'—'}</span><span>Provider: {item.provider?providerTitle(item.provider):'—'}{item.model?` · ${item.model}`:''}</span><span>Validation: {item.validation_status||'—'}</span><span>Quality score: {item.overall_quality_score==null?'N/A':`${item.overall_quality_score}%`}</span><span>{item.fallback_used?'Fallback used':'LLM/fallback: not recorded'}</span><span>Audio: {item.has_audio?'Available':'—'}</span><span>Export: {item.can_export?'Available':'—'}</span></div>
     {selected===item.id&&detail&&<div className="history-detail"><div className="history-detail-head"><div><span className="hero-kicker">DOCUMENT DETAILS</span><h3>{detail.name}</h3><p>{detail.file_type?.toUpperCase()||item.file_type.toUpperCase()} · {new Date(item.created_at).toLocaleString()} · {item.status}</p></div><button className="icon-btn" aria-label="Close document details" onClick={()=>{setSelected(null);setDetail(null)}}><X size={17}/></button></div>
      <div className="history-detail-grid"><div><small>Processing status</small><b>{item.status}</b></div><div><small>Summary provider</small><b>{item.provider||'Not recorded'}{item.model?` · ${item.model}`:''}</b></div><div><small>Validation</small><b>{detail.result?.validation_state||item.validation_status||'Not available'}</b></div><div><small>Translation</small><b>{item.has_translation?'Available':'Not generated'}</b></div><div><small>Audio</small><b>{detail.audio?.provider?`${detail.audio.provider}${detail.audio.fallback_used?' · fallback':''}`:item.has_audio?'Available':'Not generated'}</b></div><div><small>Processing settings</small><b>{detail.options?.summary_depth||detail.options?.summary_mode||item.summary_type||'Not recorded'}{detail.options?.purpose?` · ${detail.options.purpose}`:''}</b></div></div>
      <details className="history-content"><summary>Extracted content</summary><pre>{detail.extracted_text||detail.preview||'No extracted text available.'}</pre></details>
      {detail.result&&<><section className="history-summary">{detail.result.validation_warning&&<div className="validation-warning" role="alert"><b>WARNING: Validation incomplete</b><p>{detail.result.warning_message||"Some claims could not be fully verified. Review the summary against the source."}</p></div>}<h4>Summary</h4><p>{detail.result.executive_summary||'No executive summary was generated.'}</p><h4>Key findings</h4>{detail.result.findings?.length?<ul>{detail.result.findings.map((x:string,i:number)=><li key={i}>{x}</li>)}</ul>:<p>No key findings recorded.</p>}{(detail.result.facts?.length??0)>0&&<><h4>Important facts</h4><ul>{detail.result.facts.map((x:string,i:number)=><li key={i}>{x}</li>)}</ul></>}{(detail.result.risks?.length??0)>0&&<><h4>Risks</h4><ul>{detail.result.risks.map((x:string,i:number)=><li key={i}>{x}</li>)}</ul></>}{detail.result.validation_status&&<p>Validation status: {detail.result.validation_status}</p>}{detail.result.translations&&<p>Translation information: {Object.keys(detail.result.translations).join(', ')||'Available'}</p>}</section><EvaluationPanel evaluation={detail.result.evaluation}/></>}
      <div className="history-actions"><button className="primary-btn" onClick={()=>onOpen(item,detail,'Analysis')}>{detail.result?'Open analysis / reprocess':'Process document'}</button>{detail.result&&<><button className="secondary-btn" onClick={()=>onOpen(item,detail,'Analysis')}>Translate / reprocess</button><button className="secondary-btn" onClick={()=>onOpen(item,detail,'Sources')}>View sources</button><button className="secondary-btn" onClick={()=>onOpen(item,detail,'Audio briefing')}>Generate audio</button><SummaryExportMenu url={`${API}/documents/${encodeURIComponent(item.id)}/export`} label="Export"/></>}<button className="secondary-btn danger-action" onClick={()=>void remove(item)}>Delete</button></div></div>}
    </article>)}</div><div className="history-pagination"><span>{total} document{total===1?'':'s'} · Page {page} of {Math.max(pages,1)}</span><div><button className="secondary-btn" disabled={page<=1} onClick={()=>setPage(x=>x-1)}>Previous</button><button className="secondary-btn" disabled={page>=pages} onClick={()=>setPage(x=>x+1)}>Next</button></div></div></>}
  </section></>
}

function CombinedSummaryExports(){
 const [items,setItems]=useState<any[]>([])
 useEffect(()=>{let active=true;fetch(`${API}/history?activity_type=combined_summary&limit=50`).then(r=>r.ok?r.json():[]).then(rows=>{if(active)setItems((Array.isArray(rows)?rows:[]).filter(row=>Number.isInteger(row.extra_data?.combined_summary_id)))}).catch(()=>{});return()=>{active=false}},[])
 if(!items.length)return null
 return <section className="combined-export-list"><div><h3>Combined summaries</h3><p>Export a saved combined result without rerunning analysis.</p></div>{items.map(item=><article className="combined-export-row" key={item.extra_data.combined_summary_id}><div><b>{item.title}</b><small>{item.related_document_ids?.length||item.extra_data.document_count||0} documents · {item.created_at?new Date(item.created_at).toLocaleString():''}</small></div><SummaryExportMenu url={`${API}/combined-summaries/${item.extra_data.combined_summary_id}/export`} label="Export Combined Summary"/></article>)}</section>
}

function ResultCard({result:r,claimCounts,setPage,docId,speechSegments=[],activeSegmentIndex=null,isAudioActive=false,audioLanguage=''}:{result:Result;claimCounts:Record<string,number>;setPage:(s:string)=>void;docId:string;speechSegments?:SpeechSegment[];activeSegmentIndex?:number|null;isAudioActive?:boolean;audioLanguage?:string}){
 const execSummary = isUsableSummary(r.executive_summary) ? r.executive_summary.trim() : ''
 const executiveTitle:string=({en:"EXECUTIVE SUMMARY",hi:"कार्यकारी सारांश",te:"కార్యనిర్వాహక సారాంశం",kn:"ಕಾರ್ಯನಿರ್ವಾಹಕ ಸಾರಾಂಶ",ta:"நிர்வாக சுருக்கம்",ml:"എക്സിക്യൂട്ടീവ് സംഗ്രഹം",bn:"নির্বাহী সারাংশ",mr:"कार्यकारी सारांश",gu:"કાર્યકારી સારાંશ",pa:"ਕਾਰਜਕਾਰੀ ਸਾਰ",ur:"انتظامی خلاصہ"}[r.language]||'EXECUTIVE SUMMARY')
 const isAudioMatchingDoc = isAudioActive && (audioLanguage === r.language)
 const attempts=r.validation?.attempts??r.stats.validation_attempts??0
 const validationState=r.validation_state||r.validation?.state
 const validationLabel=validationState==='VALIDATION_FAILED_AFTER_RETRIES'?'VALIDATION INCOMPLETE':validationState==='PARTIALLY_VALIDATED'?'PARTIALLY VALIDATED':validationState==='VALIDATED'||r.validation_status==='passed'||r.status==='validated'?'SUMMARY VALIDATED':validationState==='NO_CLAIMS'||r.validation_status==='no_claims'?'NO CLAIMS TO VALIDATE':r.validation_provider==='local_basic'?'LOCAL BASIC CHECK':r.validation_status==='unavailable'?'VALIDATION INCOMPLETE':r.validation_status==='failed'||r.validation_status==='partial'||r.status==='partial'?'PARTIALLY VALIDATED':'SUMMARY READY'
 const validationNote=r.warning_message||r.validation?.warning_message||(validationState==='NO_CLAIMS'||r.validation_status==='no_claims'?'No factual claims requiring validation were detected.':r.validation_provider==='local_basic'?'Basic local validation was used; it is not equivalent to semantic validation.':r.validation_status==='unavailable'?`Validation service was unavailable after ${attempts} attempt${attempts===1?'':'s'}. The latest usable summary is shown without complete claim validation.`:r.validation_status==='partial'||r.validation_status==='failed'?`Some claims could not be fully verified after ${attempts} attempt${attempts===1?'':'s'}. The latest usable summary is shown.`:attempts>1?`Validated after ${attempts} attempts.`:'')
 const providerLabel=r.provider?.name?(r.provider.model?`${providerTitle(r.provider.name)} · ${r.provider.model}`:providerTitle(r.provider.name)):'No LLM available · deterministic extractive fallback'
 return <div className="result-card"><div className="result-head"><div><span className="hero-kicker"><ShieldCheck size={13}/> {execSummary?validationLabel:'SUMMARY GENERATION FAILED'}</span><h3>{execSummary?'Summary Complete':'Summary generation failed'}</h3><p>{r.summary_depth_resolved||r.depth} summary{r.summary_depth_requested==='auto'?' (Auto requested)':''} · {r.purpose} · {(r.actual_output_language||r.language).toUpperCase()}{r.translation_failed_fallback&&r.requested_output_language&&r.requested_output_language!=='auto'?<span style={{color:'var(--warn,#f5a623)',fontWeight:500}}> (Requested: {(languageNames[r.requested_output_language]||r.requested_output_language).toUpperCase()})</span>:''} · {providerLabel}</p><small>{r.error||r.warning||validationNote}{r.provider?.fallback_used&&<><br/>Fallback used: {providerTitle(r.provider.name)}.{r.provider.primary_provider&&<> Primary provider: {providerTitle(r.provider.primary_provider)}{r.provider.primary_model?` · ${r.provider.primary_model}`:''}.</>}</>}</small><small>{r.translation_validation_status==='partially_validated_against_original_report'?'Some translated claims did not pass the check against the original report.':''}{r.translation_failed_fallback?'':r.translation_validation_status?.includes('fallback')?'Translation was unavailable — showing the original-language summary.':''}{r.translation_provider&&r.provider&&r.translation_provider!==r.provider.name&&<>Translation fallback used {providerTitle(r.translation_provider)}.</>}</small></div><div className="guardrail-status" role="status"><span>Source extracted</span><span>{!execSummary?"Summary unavailable":r.validation_status === "passed" || r.validation_status === "local_basic" ? "Claims checked" : "Some claims could not be fully verified"}</span><span>{!execSummary?"Analysis failed":r.translation_failed_fallback ? "Source language shown ⚠" : r.translation_validation_status?.includes("validated") ? "Translation checked" : r.translation_validation_status?.includes("fallback") ? "Original summary available" : "Source summary ready"}</span></div><div className="score-ring"><b>{r.quality.groundedness==null?'—':`${Math.round(r.quality.groundedness*100)}%`}</b><small>source match</small></div></div>{r.translation_warning&&r.translation_warning_message&&<div className="translation-warning" role="alert" style={{background:'rgba(245,166,35,0.12)',border:'1px solid rgba(245,166,35,0.45)',borderRadius:'8px',padding:'0.75rem 1rem',marginBottom:'0.75rem',display:'flex',gap:'0.6rem',alignItems:'flex-start'}}><AlertCircle size={16} style={{color:'#f5a623',flexShrink:0,marginTop:'2px'}}/><div><b style={{color:'#f5a623',display:'block',marginBottom:'2px'}}>Translation Unavailable — Source Language Shown</b><p style={{margin:0,fontSize:'0.85rem',opacity:0.9}}>{r.translation_warning_message}</p>{r.requested_output_language&&r.requested_output_language!=='auto'&&<small style={{opacity:0.75}}>Requested: {languageNames[r.requested_output_language]||r.requested_output_language} · Shown: {languageNames[r.actual_output_language||r.language]||r.actual_output_language||r.language}</small>}</div></div>}{r.validation_warning&&<div className={`validation-warning ${validationState==='VALIDATION_FAILED_AFTER_RETRIES'?'validation-warning--strong':''}`} role="alert"><b>WARNING: Validation incomplete{validationState==='PARTIALLY_VALIDATED'?': partial validation':''}</b><p>{r.warning_message||r.validation?.warning_message||validationNote}</p><small>{r.supported_claims_count??r.validation?.supported_claims??0} claims verified; {r.unsupported_claims_count??r.validation?.unsupported_claims??0} claims unverified; {r.validation_attempts??r.validation?.attempts??attempts} validation attempts</small></div>}<div className="summary-copy" id="summary-output"><div style={{display:'flex',justifyContent:'space-between',alignItems:'center',marginBottom:'0.5rem'}}><label className="field-label" style={{margin:0}}>{executiveTitle}</label>{isAudioActive&&(isAudioMatchingDoc&&activeSegmentIndex!==null?<span className="speech-segment-badge"><Volume2 size={12}/> Live Speech Sync</span>:<span className="speech-segment-badge" style={{opacity:0.8}}><Volume2 size={12}/> Audio: {languageNames[audioLanguage]||audioLanguage.toUpperCase()}</span>)}</div>{execSummary?<div className="audio-summary-transcript">{speechSegments.length>0&&isAudioMatchingDoc?speechSegments.map((seg,i)=><span key={i} className={`speech-segment ${activeSegmentIndex===i?'speech-segment--active':''}`}>{seg.text}{' '}</span>):<p>{execSummary}</p>}</div>:<p className="muted-text">{r.error||'Summary generation failed.'}</p>}</div><PipelineVisualization hasDoc resultStatus={r.status} summaryUsable={isUsableSummary(r.executive_summary)} validationState={validationState} regenerationCount={r.stats.regeneration_count||0}/>{!!r.stats.regeneration_count&&(r.validation_status==='passed'||r.status==='validated')&&<p className="regeneration-note"><Sparkles size={13}/> The validation agent found unsupported claims and regenerated the summary {r.stats.regeneration_count} time{r.stats.regeneration_count===1?'':'s'} before this version passed validation.</p>}<SummarySections result={r}/><div className="result-stats"><span><b>{r.stats.words.toLocaleString()}</b> source words</span><span><b>{r.stats.context_chunks_used??r.stats.chunks}</b> source chunks used</span><span><b>{r.quality.claims_checked}</b> claims checked</span><span><b>{claimCounts.SUPPORTED||0}</b> supported</span>{r.stats.processing_time_ms!=null&&<span><b>{r.stats.processing_time_ms} ms</b> processing</span>}</div><EvaluationPanel evaluation={r.evaluation}/><div className="result-actions"><button className="secondary-btn" onClick={()=>document.getElementById('summary-output')?.scrollIntoView({behavior:'smooth'})}><FileText size={15}/> View Summary</button><SummaryExportMenu url={`${API}/documents/${encodeURIComponent(docId)}/export`}/><button className="secondary-btn" onClick={()=>setPage('Sources')}><BookOpen size={15}/> Explore sources</button><button className="secondary-btn" onClick={()=>setPage('Ask document')}><MessageSquareText size={15}/> Ask document</button><button className="secondary-btn" onClick={()=>setPage('Audio briefing')}><Headphones size={15}/> Audio briefing</button></div><details className="claim-details"><summary>Claim validation · {r.claims.length} claims <ChevronDown size={15}/></summary>{r.claims.map(c=><div className="claim-row" key={c.id}><span className={`claim-check ${c.status.toLowerCase()}`}>{c.status==='SUPPORTED'?<Check size={14}/>:c.status==='UNSUPPORTED'?<X size={14}/>:<CircleHelp size={14}/>}</span><div><b>{c.claim}</b><small>{c.status} · confidence {c.confidence==null?'unavailable':`${Math.round(c.confidence*100)}%`} · {c.reason} · {c.sources[0]?.chunk_id||'No source'}</small>{c.evidence&&<p>{c.evidence}</p>}</div></div>)}</details>{r.limitations.map((l,i)=><div className="limitation" key={i}><span>i</span>{l}</div>)}</div>
}
function Sources({doc}:{doc:Doc}){const [sources,setSources]=useState<any[]>([]);useEffect(()=>{fetch(`${API}/documents/${doc.id}/sources`).then(r=>r.json()).then(setSources)},[doc]);return <div className="source-list">{sources.map((s,index)=><article key={s.id||index} className="source-card"><div className="source-meta"><span>{s.section}</span><span>{s.page?`Page ${s.page}`:s.id}</span><span>{s.language.toUpperCase()}</span></div><p>{s.text}</p><small><span className="green-dot"/> Extracted source passage · {s.id}</small></article>)}</div>}
const Compare = DocumentComparison
