import * as React from 'react'
import { useEffect, useMemo, useState } from 'react'
import { useFrame } from '@react-three/fiber'
import * as THREE from 'three'
import { Activity, AlertCircle, ArrowDownToLine, AudioLines, BookOpen, Check, ChevronDown, CircleHelp, Clock3, FileText, FolderOpen, Headphones, Layers3, Menu, MessageSquareText, Network, Play, Plus, Search, ShieldCheck, Sparkles, Upload, Volume2, Waves, X } from 'lucide-react'
import { Stage, CanvasBoundary } from './components/three/shared'
import { AIProcessingPipeline, type PipelineStep } from './components/three/AIProcessingPipeline'
import { DocumentChunkVisualization } from './components/three/DocumentChunkVisualization'
import { KnowledgeGraph } from './components/three/KnowledgeGraph'
import { SourceValidationGraph } from './components/three/SourceValidationGraph'
import { RAGVisualization, type RagPhase } from './components/three/RAGVisualization'
import { AudioVisualization } from './components/three/AudioVisualization'
import { CompareVisualization } from './components/three/CompareVisualization'
import { derivePipelineSteps } from './lib/pipelineState'
import { VoiceInput, type VoiceState } from './components/VoiceInput'
import { VoiceOrb } from './components/three/VoiceOrb'

const API = 'http://127.0.0.1:8000/api'
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

type TimingItem = { index: number; start: number; end: number; text: string }
type SpeechSegment = { index: number; text: string; start: number; end: number; charStart: number; charEnd: number }

type Doc = {id:string;name:string;language:string;status:string;created_at?:string;ocr_detected?:boolean;ocr_warnings?:string[];stats:{words:number;chunks:number;pages:number|null};dataset?:{name:string;split:string;record_id:string}}
type Result = {status:string;depth:string;summary_mode?:string;summary_depth_requested?:string;summary_depth_resolved?:string;purpose:string;language:string;output_language?:string;source_language?:string;requested_language?:string;requested_output_language?:string;translation_validation_status?:string;translation_provider?:string;validation_provider?:string;validation_status?:string;executive_summary:string;findings:string[];facts?:string[];risks?:string[];decisions?:string[];recommendations?:string[];action_items?:string[];claims:{id:string;claim:string;status:string;confidence?:number|null;evidence?:string;sources:{page:number|null;section:string;chunk_id:string}[];reason:string}[];quality:{groundedness:number|null;claims_checked:number;evaluation:string};stats:{words:number;pages:number|null;estimated_tokens:number;chunks:number;context_chunks_used?:number;processing_time_ms?:number;regeneration_count?:number;validation_attempts?:number;max_validation_attempts?:number;language:string};validation?:{attempts?:number;max_attempts?:number;max_attempts_reached?:boolean};workflow?:string[];evaluation?:any;limitations:string[];provider?:{name:string;model?:string|null;fallback_used?:boolean}}
const nav = [{name:'Overview',icon:Activity},{name:'Documents',icon:FolderOpen},{name:'Analysis',icon:Layers3},{name:'Sources',icon:BookOpen},{name:'Ask document',icon:MessageSquareText},{name:'Compare',icon:Network},{name:'Audio briefing',icon:Headphones}]
const customDetailOptions = [{value:'quick',label:'Concise',description:'Short and focused'},{value:'standard',label:'Balanced',description:'Coverage with useful detail'},{value:'deep',label:'Detailed',description:'Maximum context'}] as const
const customSectionOptions = [{value:'findings',label:'Key findings'},{value:'facts',label:'Important facts'},{value:'risks',label:'Risks'},{value:'decisions',label:'Decisions'},{value:'recommendations',label:'Recommendations'},{value:'action_items',label:'Action items'}] as const

function Orb(){
 const ref=React.useRef<THREE.Mesh>(null!)
 useFrame(({clock})=>{if(ref.current){ref.current.rotation.y=clock.elapsedTime*.08;ref.current.rotation.x=Math.sin(clock.elapsedTime*.16)*.08}})
 return <><mesh ref={ref}><icosahedronGeometry args={[1.23,2]}/><meshStandardMaterial color="#19342b" wireframe emissive="#54dfa0" emissiveIntensity={.72}/></mesh><mesh scale={.72}><icosahedronGeometry args={[1.23,2]}/><meshBasicMaterial color="#62e3a1" wireframe transparent opacity={.18}/></mesh><pointLight color="#5df0a2" intensity={9} distance={6}/></>
}
const visualFallback=<div className="canvas-fallback">Document → Intelligence → Validated summary → Audio</div>
function Visual(){return <div className="visual"><Stage height="100%" fallback={visualFallback} camera={{position:[0,0,4.5]}}><ambientLight intensity={1.2}/><Orb/></Stage><div className="orbit-label label-doc">DOCUMENT</div><div className="orbit-label label-ai">INTELLIGENCE</div><div className="orbit-label label-audio">BRIEFING</div><div className="visual-bottom">SOURCE-GROUNDED BY DESIGN <span>●</span></div></div>}

export default function App(){
 const [page,setPage]=useState('Overview'),[docs,setDocs]=useState<Doc[]>([]),[doc,setDoc]=useState<Doc|null>(null),[result,setResult]=useState<Result|null>(null),[depth,setDepth]=useState('standard'),[customDetail,setCustomDetail]=useState('standard'),[customSections,setCustomSections]=useState<string[]>(['findings','facts','risks','decisions','recommendations','action_items']),[purpose,setPurpose]=useState('Executive briefing'),[language,setLanguage]=useState('auto'),[ocrLanguage,setOcrLanguage]=useState('auto'),[audioLanguage,setAudioLanguage]=useState('auto'),[audience,setAudience]=useState('Leadership'),[busy,setBusy]=useState(false),[error,setError]=useState(''),[question,setQuestion]=useState(''),[answer,setAnswer]=useState<any>(null),[toast,setToast]=useState(''),[previewOpen,setPreviewOpen]=useState(false),[audioSpeed,setAudioSpeed]=useState(1),[selectedVoice,setSelectedVoice]=useState(''),[voiceOptions,setVoiceOptions]=useState<SpeechSynthesisVoice[]>([]),[audioPaused,setAudioPaused]=useState(false),[audioUrl,setAudioUrl]=useState(''),[audioBusy,setAudioBusy]=useState(false),[askBusy,setAskBusy]=useState(false),[selectedChunk,setSelectedChunk]=useState<string|null>(null),[speaking,setSpeaking]=useState(false)
 const [audioNarrationText,setAudioNarrationText]=useState('')
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
 const refresh=()=>fetch(`${API}/documents`).then(r=>r.json()).then(setDocs).catch(()=>{})
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

  const pattern=/[^.!?।॥۔؟\n]+(?:[.!?।॥۔؟]+|\n+|$)/g
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

 async function upload(file:File){setBusy(true);setError('');setToast('Scanning for image text with local OCR when needed…');try{const fd=new FormData();fd.append('file',file);fd.append('ocr_language',ocrLanguage);const r=await fetch(`${API}/documents/upload`,{method:'POST',body:fd});const data=await r.json();if(!r.ok)throw Error(data.detail||'Upload failed');setDoc(data);setResult(null);setPage('Analysis');notify(data.ocr_detected?'Scanned/image content detected — OCR text is ready for analysis.':'Document extracted and ready for analysis');refresh()}catch(e:any){setError(e.message)}finally{setBusy(false)}}
 async function demo(){setBusy(true);setError('');try{const r=await fetch(`${API}/demo`,{method:'POST'});const data=await r.json();if(!r.ok)throw Error(data.detail||'Demo unavailable');setDoc(data);setResult(data.result);setPage('Analysis');refresh();notify('Demo document analyzed · deterministic, source-grounded sample')}catch(e:any){setError(e.message)}finally{setBusy(false)}}
 async function analyze(){
  if(!doc)return
  const targetLanguage=language==='auto'?doc.language:language
  setBusy(true);setError('');speechSynthesis.cancel();if(audioRef.current)audioRef.current.pause();setAudioUrl('');setAudioNarrationText('');setAudioTimings([]);setActiveSegmentIndex(null);setSpeaking(false);setAudioPaused(false)
  try{
   const endpoint=result&&targetLanguage!==result.language?'translate':'process'
   const r=await fetch(`${API}/documents/${doc.id}/${endpoint}`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({summary_depth:depth,custom_detail:customDetail,custom_sections:customSections,purpose,source_language:doc.language,output_language:language,audience})})
   const data=await r.json();if(!r.ok)throw Error(data.detail||'Analysis failed')
   let finalResult=data
   if(doc.dataset){try{const evalRes=await fetch(`${API}/evaluation/${doc.id}`);const evalData=await evalRes.json();if(evalRes.ok&&evalData?.metrics)finalResult={...data,evaluation:evalData}}catch{}}
   setResult(finalResult)
   if(finalResult.language!==targetLanguage){notify(`Translation to ${languageNames[targetLanguage]||targetLanguage.toUpperCase()} failed; showing the source-language summary in ${languageNames[finalResult.language]||finalResult.language.toUpperCase()}.`)}
   else if(finalResult.language!==finalResult.source_language){notify(`Translation to ${languageNames[finalResult.language]||finalResult.language.toUpperCase()} complete`)}
   else{notify('Analysis complete ? summary language follows the document')}
  }catch(e:any){setError(e.message);notify('Analysis error: '+e.message)}finally{setBusy(false)}
 }
 async function ask(){if(!doc||!question.trim())return;setAskBusy(true);setSelectedChunk(null);try{const r=await fetch(`${API}/documents/${doc.id}/ask`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question,output_language:language})});setAnswer(await r.json())}catch{setError('Could not reach the document service.')}finally{setAskBusy(false)}}
 async function play(){
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
     rate:audioSpeed
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
    let currentNarration=result.executive_summary;
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

   const narration=d.text||result.executive_summary;
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
 const stats=result?.stats
 const claimCounts=useMemo(()=>result?.claims.reduce((a,c)=>(a[c.status]=(a[c.status]||0)+1,a),{} as Record<string,number>)||{},[result])
 const overviewSteps=useMemo<PipelineStep[]>(()=>derivePipelineSteps({hasDoc:!!doc,busy,error:!!error,resultStatus:result?.status??null,regenerationCount:result?.stats.regeneration_count||0,audioActive:!!audioUrl}),[doc,busy,error,result,audioUrl])
 return <div className="app-shell"><aside className="sidebar"><div className="brand"><div className="brand-mark"><Sparkles size={18}/></div><span>scribe<span className="brand-dot">.</span></span><button className="icon-btn mobile-menu" aria-label="Open menu"><Menu/></button></div><div className="workspace-label">WORKSPACE <ChevronDown size={12}/></div><div className="workspace"><div className="workspace-icon">N</div><div><b>Northstar Studio</b><small>Private workspace</small></div><ChevronDown size={15}/></div><div className="nav-label">WORKSPACE</div><nav>{nav.map(({name,icon:Icon})=><button key={name} className={`nav-item ${page===name?'active':''}`} onClick={()=>setPage(name)}><Icon size={17}/>{name}{name==='Documents'&&docs.length>0&&<small className="nav-count">{docs.length}</small>}</button>)}</nav><div className="side-spacer"/><div className="plan-card"><div className="plan-icon"><Sparkles size={15}/></div><b>Local demo mode</b><p>Source-grounded analysis is ready. AI providers can be connected as you scale.</p><div className="plan-meter"><i/></div><small>0 / 5 documents today</small></div><button className="profile"><div className="avatar">GS</div><div><b>Guest session</b><small>Local workspace</small></div><ChevronDown size={15}/></button></aside>
  <main className="main"><header className="topbar"><div className="breadcrumbs"><span>Workspace</span><span className="crumb-slash">/</span><b>{page}</b>{doc&&<><span className="crumb-slash">/</span><span>{doc.name}</span></>}</div><div className="top-actions"><div className="system-pill"><span/> All systems operational</div><button className="icon-btn" aria-label="Help"><CircleHelp size={18}/></button><div className="avatar avatar-sm">GS</div></div></header>
  <div className="content">
  {page==='Overview'&&<><div className="welcome-row"><div><div className="eyebrow"><span/> DOCUMENT INTELLIGENCE, REIMAGINED</div><h1>Good morning, Guest <span className="wave">✳</span></h1><p className="subhead">Turn complex documents into clear, trusted understanding.</p></div><div className="welcome-actions"><label className="overview-language"><span>SUMMARY LANGUAGE</span><select aria-label="Summary language" value={language} onChange={e=>setLanguage(e.target.value)}>{outputLanguages.map(l=><option key={l.code} value={l.code}>{l.name}</option>)}</select></label><button className="secondary-btn" onClick={demo}><Sparkles size={15}/> Try demo</button><button className="primary-btn" onClick={()=>document.getElementById('file-input')?.click()}><Plus size={17}/> New analysis</button></div></div><div className="hero-card"><div className="hero-copy"><span className="hero-kicker"><Sparkles size={13}/> YOUR INTELLIGENCE WORKSPACE</span><h2>Turn long documents<br/>into <em>trusted intelligence.</em></h2><p>Analyze, validate, and explore your documents with source-grounded intelligence. Every insight traces back to its source.</p><button className="hero-cta" onClick={()=>document.getElementById('file-input')?.click()}>Analyze a document <span>→</span></button><div className="hero-foot"><span><ShieldCheck size={15}/> Evidence, always in view</span><i/> <span><Waves size={15}/> Audio from validated summaries</span></div></div><Visual/></div><div className="section-title-row"><div><h3>Your workspace</h3><p>A clear view of your document activity.</p></div><button className="text-btn" onClick={()=>setPage('Documents')}>View all documents <span>→</span></button></div><div className="metric-grid"><Metric icon={<FileText/>} label="Documents analyzed" value={String(docs.length).padStart(2,'0')} change="This session"/><Metric icon={<ShieldCheck/>} label="Grounded claims" value={String(result?.quality.claims_checked||0).padStart(2,'0')} change="Source checked"/><Metric icon={<AudioLines/>} label="Audio briefings" value="00" change="Browser playback"/><Metric icon={<Clock3/>} label="Avg. processing" value="—" change="Local mode"/></div><div className="section-title-row compact"><div><h3>AI processing pipeline</h3><p>Live status of your current document, from extraction through validated audio.</p></div></div><AIProcessingPipeline steps={overviewSteps}/><div className="section-title-row compact"><div><h3>Recent documents</h3><p>Your latest work, all in one place.</p></div><button className="text-btn" onClick={()=>setPage('Documents')}>Browse documents <span>→</span></button></div><DocumentList docs={docs} open={d=>{setDoc(d);setResult(null);setPage('Analysis')}} emptyAction={()=>document.getElementById('file-input')?.click()}/></>}
  {page==='Documents'&&<><PageHeading title="Your documents" subtitle="Manage uploaded files and load a bounded GovReport sample." action={<button className="primary-btn" onClick={()=>document.getElementById('file-input')?.click()}><Plus size={17}/> Add document</button>}/>{doc&&<div className="chunk-visual-card"><div className="card-head"><div><h3>{doc.name}</h3><p>Document → Extraction → Chunking → Embeddings, sized to this document's real chunk count.</p></div></div><DocumentChunkVisualization status="extracted" chunkCount={doc.stats.chunks} wordCount={doc.stats.words} pages={doc.stats.pages}/></div>}<GovReportBrowser onLoaded={d=>{setDoc(d);setResult(null);setPage('Analysis')}}/><DocumentList docs={docs} open={d=>{setDoc(d);setResult(null);setPage('Analysis')}} emptyAction={()=>document.getElementById('file-input')?.click()}/></>}
  {page==='Analysis'&&<><PageHeading title="Document analysis" subtitle={doc?`Configure an analysis for ${doc.name}.`:'Choose a document to begin.'}/>{doc?<><div className="file-card"><div className="file-icon"><FileText/></div><div className="file-info"><b>{doc.name}</b><small>{doc.stats.words.toLocaleString()} words · {doc.stats.chunks} evidence chunks · Detected: {languageNames[doc.language]||doc.language.toUpperCase()}</small></div><span className="status-chip"><span/> Extracted</span><button className="icon-btn" aria-label="Remove document" onClick={()=>{setDoc(null);setResult(null)}}><X size={17}/></button></div><div className="lang-banner"><div className="lang-banner-item"><span className="field-label" style={{marginBottom:0}}>SOURCE LANGUAGE</span><b className="lang-badge">{doc.language==='en'?'🇬🇧 English':`🇮🇳 ${languageNames[doc.language]||doc.language.toUpperCase()}`}</b></div><div className="lang-arrow">→</div><div className="lang-banner-item"><span className="field-label" style={{marginBottom:0}}>OUTPUT LANGUAGE</span><b className="lang-badge lang-badge--output">{language==='auto'?`Auto / ${languageNames[doc.language]||doc.language.toUpperCase()}`:languageLabel(language)}</b></div>{result&&result.source_language&&result.language!==result.source_language&&!(result&&(language==='auto'?doc.language:language)!==result.language)&&<span className="translation-chip">✓ Translation validated against original</span>}{result&&(language==='auto'?doc.language:language)!==result.language&&<span className="translation-chip" style={{background:'rgba(234, 179, 8, 0.15)',color:'#fbbf24',borderColor:'rgba(234, 179, 8, 0.4)'}}>⚠️ Pending translation · Click "Translate document" below</span>}</div>{doc.ocr_detected&&<div className="ocr-notice">Scanned or image content detected. Text was extracted for this analysis.</div>}{doc.ocr_warnings?.map((w,i)=><div className="ocr-notice" key={i}>{w}</div>)}<div className="analysis-layout"><div className="config-card"><div className="card-head"><div><h3>Shape your summary</h3><p>Choose what matters to you. You can change language without re-uploading.</p></div><Sparkles className="muted-icon" size={19}/></div><PipelineVisualization hasDoc busy={busy} error={!!error}/><label className="field-label">SUMMARY DEPTH</label><div className="depth-grid" role="radiogroup" aria-label="Summary depth">{[['quick','Quick','A concise overview'],['standard','Standard','Balanced coverage'],['deep','Deep','Detailed analysis'],['auto','Auto','Recommended mode'],['custom','Custom','Choose sections and detail']].map(([v,n,d])=><button type="button" role="radio" aria-checked={depth===v} className={`depth-option ${depth===v?'selected':''}`} onClick={()=>setDepth(v)} key={v}><span className="radio" aria-hidden="true"/><b>{n}</b><small>{d}</small></button>)}</div>{depth==='custom'&&<section className="custom-summary" aria-labelledby="custom-summary-title"><header className="custom-summary-head"><div><span className="custom-eyebrow">YOUR CONFIGURATION</span><h4 id="custom-summary-title">Custom Summary</h4><p>Choose how detailed your summary should be and which sections to include.</p></div><span className="custom-count" aria-live="polite">{customSections.length} of {customSectionOptions.length} selected</span></header><section className="custom-control-group" aria-labelledby="custom-detail-heading"><div className="custom-group-heading"><div><h5 id="custom-detail-heading">Detail level</h5><p>Set the amount of context in your summary.</p></div></div><div className="custom-detail-options" role="group" aria-labelledby="custom-detail-heading">{customDetailOptions.map(option=><button type="button" key={option.value} className={`custom-detail-option ${customDetail===option.value?'selected':''}`} aria-pressed={customDetail===option.value} onClick={()=>setCustomDetail(option.value)}><span className="custom-detail-indicator" aria-hidden="true"/><span><b>{option.label}</b><small>{option.description}</small></span></button>)}</div></section><section className="custom-control-group custom-sections-group" aria-labelledby="custom-sections-heading"><div className="custom-group-heading custom-sections-heading"><div><h5 id="custom-sections-heading">Include sections</h5><p>Select the sections you want in the generated summary.</p></div><div className="custom-section-actions"><button type="button" className="text-btn" onClick={()=>setCustomSections(customSectionOptions.map(option=>option.value))} disabled={customSections.length===customSectionOptions.length}>Select all</button><span aria-hidden="true">?</span><button type="button" className="text-btn" onClick={()=>setCustomSections([])} disabled={customSections.length===0}>Clear all</button></div></div><div className="custom-sections" role="group" aria-labelledby="custom-sections-heading">{customSectionOptions.map(option=>{const selected=customSections.includes(option.value);return <button type="button" role="checkbox" aria-checked={selected} key={option.value} className={`custom-section-card ${selected?'selected':''}`} onClick={()=>setCustomSections(current=>selected?current.filter(value=>value!==option.value):customSectionOptions.filter(item=>item.value===option.value||current.includes(item.value)).map(item=>item.value))}><span className="custom-section-check" aria-hidden="true">{selected&&<Check size={13}/>}</span><span>{option.label}</span></button>})}</div></section><div className="custom-summary-preview" aria-live="polite"><span className="custom-preview-mark"><Sparkles size={15}/></span><span><b>Custom summary</b><small>{customDetailOptions.find(option=>option.value===customDetail)?.label} detail ? {customSections.length} section{customSections.length===1?'':'s'} selected{customSections.length===0?' ? executive overview only':''}</small></span></div></section>}<div className="field-grid"><label><span className="field-label">PURPOSE</span><select value={purpose} onChange={e=>setPurpose(e.target.value)}>{['Executive briefing','General understanding','Research','Decision making','Risk analysis','Action planning','Presentation preparation'].map(x=><option key={x}>{x}</option>)}</select></label><label><span className="field-label">AUDIENCE</span><select value={audience} onChange={e=>setAudience(e.target.value)}>{['Leadership','Research team','Project team','General audience','Technical audience'].map(x=><option key={x}>{x}</option>)}</select></label><label><span className="field-label">OUTPUT LANGUAGE</span><select value={language} onChange={e=>setLanguage(e.target.value)}>{outputLanguages.map(l=><option key={l.code} value={l.code}>{l.name}</option>)}</select></label></div><div className="config-note"><ShieldCheck size={16}/><span><b>Validation first.</b> Claims are checked against extracted source text before they appear as validated or become eligible for audio.</span></div>{result&&(language==='auto'?doc.language:language)!==result.language&&<div className="config-note" style={{borderLeftColor:'#f59e0b',background:'rgba(245, 158, 11, 0.08)'}}><Sparkles size={16} color="#f59e0b"/><span><b>Translation ready.</b> Target language set to <b>{languageLabel(language==='auto'?doc.language:language)}</b>. Click <b>"Translate document"</b> below to translate and validate against the source.</span></div>}{error&&<div className="config-note" style={{borderLeftColor:'#ef4444',background:'rgba(239, 68, 68, 0.1)',color:'#fca5a5'}}><AlertCircle size={16} color="#ef4444"/><span><b>Notice:</b> {error}</span></div>}<button className="primary-btn analyze-btn" disabled={busy} onClick={analyze}>{busy?<><span className="spinner"/> {result&&(language==='auto'?doc.language:language)!==result.language?'Translating…':'Analyzing…'}</>:result&&(language==='auto'?doc.language:language)!==result.language?<><Sparkles size={16}/> Translate document to {languageLabel(language==='auto'?doc.language:language)} <span>→</span></>:<><Sparkles size={16}/> Analyze document <span>→</span></>}</button></div><div className="side-info-card"><div className="side-info-icon"><Layers3 size={19}/></div><h3>Document profile</h3><p>Extraction metadata from your uploaded file.</p><div className="profile-row"><span>Word count</span><b>{doc.stats.words.toLocaleString()}</b></div><div className="profile-row"><span>Evidence chunks</span><b>{doc.stats.chunks}</b></div><div className="profile-row"><span>Detected language</span><b>{languageNames[doc.language]||doc.language.toUpperCase()}</b></div><div className="profile-row"><span>File type</span><b>{doc.name.split('.').pop()?.toUpperCase()}</b></div><div className="provider-note"><span className="amber-dot"/> <span>Sarvam AI primary<br/><small>OpenRouter secondary ? local fallback available</small></span></div></div></div></>:<div className="empty-card"><div className="empty-mark"><Upload/></div><h3>Start with a document</h3><p>Upload a PDF, DOCX, TXT, JSON, JPG, or PNG file to extract text and configure an analysis.</p><button className="primary-btn" onClick={()=>document.getElementById('file-input')?.click()}><Upload size={16}/> Choose a document</button></div>}{result&&<ResultCard result={result} claimCounts={claimCounts} setPage={setPage} docId={doc?.id||''} speechSegments={speechSegments} activeSegmentIndex={activeSegmentIndex} isAudioActive={!audioPaused&&(!!audioUrl||speaking)} audioLanguage={audioLanguage}/>}</>}
 {page==='Sources'&&<><PageHeading title="Source library" subtitle="Browse extracted passages and trace claims back to their source."/>{doc?<>{result&&<div className="source-graph-card"><div className="card-head"><div><h3>Claim → evidence graph</h3><p>Each validated claim traced back to its source passage. Click a node for detail.</p></div></div><SourceValidationGraph claims={result.claims}/></div>}<Sources doc={doc}/></>:<EmptySelect docs={docs} choose={d=>{setDoc(d);setPage('Sources')}}/>}</>}
 {page==='Ask document'&&<><PageHeading title="Ask your document" subtitle="Search extracted passages with lexical retrieval. Answers stay close to the source."/>{doc?<div className="chat-card"><div className="chat-context"><div className="file-icon small-file"><FileText size={16}/></div><span><b>{doc.name}</b><small>{doc.stats.chunks} searchable chunks</small></span><span className="online-tag"><i/> Source search ready</span>{(language==='auto'?doc.language:language)!=='en'&&<span className="lang-answer-tag">Answers in {languageLabel(language==='auto'?doc.language:language)}</span>}</div><div className="chat-empty"><div className="chat-orb"><MessageSquareText/></div><h3>What would you like to know?</h3><p>Ask about the document. Scribe will surface relevant extracted passages with citations.</p><div className="suggestions">{['What are the key findings?','What risks are mentioned?','What recommendations are included?'].map(s=><button key={s} onClick={()=>setQuestion(s)}>{s} <span>↗</span></button>)}</div>{(askBusy||answer)&&<RAGVisualization phase={askBusy?'retrieving':answer?'answered':'idle'} chunks={answer?.sources||[]} selectedChunk={selectedChunk} onSelectChunk={id=>setSelectedChunk(current=>current===id?null:id)}/>}{answer&&<div className="answer"><div className="answer-label"><Sparkles size={14}/> SOURCE RESPONSE <span className="grounded-tag">{answer.grounded?'Evidence found':'No evidence'}</span></div><p>{answer.answer}</p>{answer.sources?.map((s:any)=><details className="answer-source" key={s.chunk_id} open={selectedChunk===s.chunk_id} onToggle={e=>{const isOpen=(e.target as HTMLDetailsElement).open;setSelectedChunk(current=>isOpen?s.chunk_id:(current===s.chunk_id?null:current))}}><summary>{s.section} ? {s.page?`Page ${s.page}`:s.chunk_id} ? Open source passage</summary><p>{s.text}</p></details>)}</div>}</div><form className="ask-form" onSubmit={e=>{e.preventDefault();ask()}}><input value={question} onChange={e=>setQuestion(e.target.value)} placeholder="Ask a question about this document…"/><button className="primary-btn" type="submit">Ask <span>↑</span></button></form></div>:<EmptySelect docs={docs} choose={d=>{setDoc(d);setPage('Ask document')}}/>}</>}
  {page==='Audio briefing'&&<><PageHeading title="Audio briefing" subtitle="Choose a narration language. The summary can be translated for speech."/>{doc&&result?<div className="audio-card"><div className="audio-glow"><AudioLines size={30}/></div><div className="audio-content"><span className="hero-kicker"><ShieldCheck size={13}/> {result.status==='validated'?'VALIDATED SUMMARY':result.status==='partial'?'PARTIALLY VALIDATED SUMMARY':'SUMMARY READY'}</span><h2>Your document, <em>in audio.</em></h2>{currentPlayingProvider&&<div className="speech-segment-badge" style={{marginBottom:'0.75rem',width:'fit-content'}}><Volume2 size={13}/><span>TTS: <b>{currentPlayingProvider}</b> {currentPlayingVoice?`(${currentPlayingVoice})`:''}</span></div>}<div className="audio-summary-transcript">{speechSegments.length>0?speechSegments.map((seg,i)=><span key={i} className={`speech-segment ${activeSegmentIndex===i?'speech-segment--active':''}`}>{seg.text}{' '}</span>):<p>{result.executive_summary||'No summary text is available.'}</p>}</div>{(audioUrl||speaking)&&<AudioVisualization audioEl={audioRef.current} playing={!audioPaused&&(!!audioUrl||speaking)}/>}<div className="audio-controls"><button className="play-btn" disabled={audioBusy} onClick={play}>{audioBusy?<><span className="spinner"/> Generating…</>:<><Play size={17} fill="currentColor"/> {selectedVoice.startsWith('edge:')?'Generate & play':'Play briefing'}</>}</button><label>LANGUAGE<select value={audioLanguage} onChange={e=>{const newLang=e.target.value;setAudioLanguage(newLang);setAudioUrl('');speechSynthesis.cancel();setSpeaking(false);setActiveSegmentIndex(null);const sysVoice=voiceOptions.find(v=>v.lang.toLowerCase().startsWith(newLang));if(sysVoice){setSelectedVoice(`browser:${sysVoice.voiceURI}`)}else{const neural=(neuralVoices[newLang]||[])[0];if(neural)setSelectedVoice(`edge:${neural.id}`)}}}>{supportedLanguages.map(l=><option key={l.code} value={l.code}>{l.name}</option>)}</select></label><label>VOICE<select value={selectedVoice} onChange={e=>{setSelectedVoice(e.target.value);setAudioUrl('');speechSynthesis.cancel();setSpeaking(false);setActiveSegmentIndex(null)}}>{voiceOptions.filter(v=>v.lang.toLowerCase().startsWith(audioLanguage)).map(v=><option key={v.voiceURI} value={`browser:${v.voiceURI}`}>{v.name} · system</option>)}<optgroup label="Online neural voices">{(neuralVoices[audioLanguage]||[]).map(v=><option key={v.id} value={`edge:${v.id}`}>{v.name}</option>)}</optgroup></select></label><label>SPEED<input type="range" min="0.7" max="1.3" step="0.1" value={audioSpeed} onChange={e=>{const rate=Number(e.target.value);setAudioSpeed(rate);if(audioRef.current)audioRef.current.playbackRate=rate}}/></label>{audioUrl?<a className="secondary-btn audio-download" href={audioUrl} download={`scribe-briefing-${audioLanguage}.mp3`}><ArrowDownToLine size={14}/> Download</a>:selectedVoice.startsWith('browser:')?<button className="secondary-btn" onClick={()=>{if(audioPaused){speechSynthesis.resume();setAudioPaused(false)}else{speechSynthesis.pause();setAudioPaused(true)}}}>{audioPaused?'Resume':'Pause'}</button>:null}</div>{audioUrl&&<audio ref={audioRef} className="generated-audio" controls src={audioUrl} onTimeUpdate={syncAudioTime} onSeeking={syncAudioTime} onSeeked={syncAudioTime} onPlay={()=>setAudioPaused(false)} onPause={()=>setAudioPaused(true)} onEnded={()=>{setAudioPaused(false);setActiveSegmentIndex(null)}} />}{selectedVoice.startsWith('edge:')?<small className="audio-disclaimer">Online neural speech sends only this summary to Microsoft’s speech service to generate audio. Audio is held in this browser session.</small>:<small className="audio-disclaimer">System voices run on your device. Translation uses your configured OpenRouter provider when narration and summary languages differ. If a matching system voice is unavailable, choose an online neural voice.</small>}</div></div>:<EmptySelect docs={docs} choose={d=>{setDoc(d);setPage('Analysis')}} label="Choose a processed document to create an audio briefing."/>}</>}
 {page==='Compare'&&<Compare/>}
 </div></main><label className="ocr-language-picker"><span>OCR INPUT LANGUAGE</span><select aria-label="OCR input language" value={ocrLanguage} onChange={e=>setOcrLanguage(e.target.value)}>{[{code:"auto",name:"Auto detect"},...supportedLanguages.slice(0,5)].map(l=><option key={l.code} value={l.code}>{l.name}</option>)}</select></label><input id="file-input" className="hidden" type="file" accept=".pdf,.docx,.txt,.json,.jpg,.jpeg,.png" onChange={e=>{const f=e.target.files?.[0];if(f)upload(f);e.currentTarget.value=''}}/>{busy&&<div className="processing-toast"><span className="spinner"/> Extracting text, scanning images, and running OCR…</div>}{error&&<div className="error-toast"><span>{error}</span><button aria-label="Dismiss error" onClick={()=>setError('')}><X size={16}/></button></div>}{toast&&<div className="success-toast"><Check size={16}/>{toast}</div>}{previewOpen&&<button onClick={()=>setPreviewOpen(false)} className="hidden"/>}</div>
}

function Metric({icon,label,value,change}:{icon:any;label:string;value:string;change:string}){return <div className="metric-card"><div className="metric-top">{icon}<span>{change}</span></div><b>{value}</b><small>{label}</small></div>}
function PageHeading({title,subtitle,action}:{title:string;subtitle:string;action?:any}){return <div className="page-heading"><div><div className="eyebrow"><span/> SOURCE-GROUNDED INTELLIGENCE</div><h1>{title}</h1><p>{subtitle}</p></div>{action}</div>}
function DocumentList({docs,open,emptyAction}:{docs:Doc[];open:(d:Doc)=>void;emptyAction?:()=>void}){return docs.length?<div className="doc-list">{docs.map((d,i)=><button className="doc-row" key={d.id} onClick={()=>open(d)}><div className="file-icon"><FileText/></div><div className="doc-row-main"><b>{d.name}</b><small>{d.stats.words.toLocaleString()} words · {d.stats.chunks} chunks · Added {new Date(d.created_at||Date.now()).toLocaleDateString()}</small></div><span className="language-chip">{d.language.toUpperCase()}</span><span className="status-chip"><span/> {d.status.toLowerCase()}</span><span className="row-arrow">↗</span></button>)}</div>:<div className="empty-card"><div className="empty-mark"><FileText/></div><h3>Your workspace is ready</h3><p>Upload your first document to extract content, check claims, and explore the evidence.</p><button className="primary-btn" onClick={emptyAction}><Upload size={16}/> Upload a document</button></div>}
function EmptySelect({docs,choose,label='Select a document to continue.'}:{docs:Doc[];choose:(d:Doc)=>void;label?:string}){return <div className="empty-card"><div className="empty-mark"><FolderOpen/></div><h3>Choose a document</h3><p>{label}</p>{docs.length?docs.map(d=><button key={d.id} className="select-doc" onClick={()=>choose(d)}>{d.name}<span>→</span></button>):<small>Upload one from the Documents page first.</small>}</div>}
function GovReportBrowser({onLoaded}:{onLoaded:(doc:Doc)=>void}){
 const [split,setSplit]=useState('train'),[limit,setLimit]=useState(20),[evaluationMode,setEvaluationMode]=useState(false),[records,setRecords]=useState<any[]>([]),[busy,setBusy]=useState(false),[message,setMessage]=useState('')
 async function browse(){setBusy(true);setMessage('');try{const r=await fetch(`${API}/dataset/govreport?split=${split}&limit=${limit}&evaluation_mode=${evaluationMode}`);const data=await r.json();if(!r.ok)throw Error(data.detail||'Could not load GovReport');setRecords(data.records);if(!data.records.length)setMessage('No records returned for this split.')}catch(e:any){setMessage(e.message)}finally{setBusy(false)}}
 async function load(recordId:string){setBusy(true);setMessage('');try{const r=await fetch(`${API}/dataset/govreport/${encodeURIComponent(recordId)}/load?split=${split}&limit=${limit}&evaluation_mode=${evaluationMode}`,{method:'POST'});const data=await r.json();if(!r.ok)throw Error(data.detail||'Could not load selected report');onLoaded(data)}catch(e:any){setMessage(e.message)}finally{setBusy(false)}}
 return <section className="dataset-picker">
  <div className="dataset-hero">
   <div className="dataset-hero-icon"><BookOpen size={20}/></div>
   <div className="dataset-hero-copy"><div className="dataset-kicker">HUGGING FACE DATASET <span>·</span> GOVREPORT</div><h2>Explore government reports</h2><p>Browse a bounded sample, then analyze a selected report. Human-written reference summaries are kept for evaluation only.</p></div>
   <div className="dataset-splits"><span>TRAIN</span><span>VALIDATION</span><span>TEST</span></div>
  </div>
  <div className="dataset-browser">
   <div className="dataset-browser-title"><div><h3>Find a report</h3><p>Choose a split and sample size to preview available documents.</p></div>{records.length>0&&<span className="dataset-count">{records.length} reports loaded</span>}</div>
   <div className="dataset-controls">
    <label><span>DATA SPLIT</span><select value={split} onChange={e=>setSplit(e.target.value)}><option value="train">Training set</option><option value="validation">Validation set</option><option value="test">Test set</option></select></label>
    <label><span>SAMPLE SIZE</span><input type="number" min="1" max={evaluationMode?50000:100} value={limit} onChange={e=>setLimit(Number(e.target.value))}/></label>
    <label className="dataset-eval-toggle"><input type="checkbox" checked={evaluationMode} onChange={e=>{setEvaluationMode(e.target.checked);setLimit(e.target.checked?500:20)}}/><span><b>Evaluation subset</b><small>Allow a larger bounded sample</small></span></label>
    <button className="primary-btn dataset-browse-btn" disabled={busy} onClick={browse}>{busy?<><span className="spinner"/> Loading sample</>:<>Browse reports <span>→</span></>}</button>
   </div>
   {message&&<p className="form-message">{message}</p>}
   {records.length>0?<div className="dataset-records">{records.map((row,index)=><article className="dataset-record" key={row.id}><div className="dataset-record-index">{String(index+1).padStart(2,'0')}</div><div className="dataset-record-body"><div className="dataset-record-heading"><b>GovReport record</b><span>{row.id_source==='content_hash_fallback'?'Content ID':'Dataset ID'} · {row.id}</span></div><small>{row.report_characters.toLocaleString()} report characters <i/> {row.summary_characters.toLocaleString()} reference characters</small><p>{row.preview}…</p></div><button className="dataset-load-btn" disabled={busy} onClick={()=>load(row.id)}>Load report <span>→</span></button></article>)}</div>:<div className="dataset-empty"><div className="dataset-empty-icon"><BookOpen size={18}/></div><div><b>Your sample will appear here</b><p>Browse reports to preview the source text and load one into your workspace.</p></div></div>}
   <div className="dataset-footnote"><span>i</span><p>Development browsing is capped at 100 rows. Use the test split for final evaluation; reference summaries are never used as source evidence.</p></div>
  </div>
 </section>
}

function SummarySections({result:r}:{result:Result}){
 const sectionNames:Record<string,string[]>={
  en:['KEY FINDINGS','IMPORTANT FACTS','RISKS','DECISIONS','RECOMMENDATIONS','ACTION ITEMS'],
  hi:['मुख्य निष्कर्ष','महत्वपूर्ण तथ्य','जोखिम','निर्णय','सिफारिशें','कार्रवाई के बिंदु'],
  te:['ముఖ్యమైన ఫలితాలు','ముఖ్యమైన వాస్తవాలు','ప్రమాదాలు','నిర్ణయాలు','సిఫార్సులు','చర్య అంశాలు'],
  kn:['ಮುಖ್ಯಾಂಶಗಳು','ಪ್ರಮುಖ ಸಂಗತಿಗಳು','ಅಪಾಯಗಳು','ನಿರ್ಧಾರಗಳು','ಶಿಫಾರಸುಗಳು','ಕಾರ್ಯ ಕ್ರಮಗಳು'],
  ta:['முக்கியக் கண்டறிதல்கள்','முக்கிய உண்மைகள்','அபாயங்கள்','முடிவுகள்','பரிந்துரைகள்','செயல் உருப்படிகள்'],
  ml:['പ്രധാന കണ്ടെത്തലുകൾ','പ്രധാന വസ്തുതകൾ','അപകടസാധ്യതകൾ','തീരുമാനങ്ങൾ','ശുപാർശകൾ','നടപടി കാര്യങ്ങൾ'],
  bn:['মূল অনুসন্ধান','গুরুত্বপূর্ণ তথ্য','ঝুঁকি','সিদ্ধান্ত','সুপারিশ','করণীয়'],
  mr:['मुख्य निष्कर्ष','महत्त्वाची तथ्ये','जोखीम','निर्णय','शिफारसी','कृती मुद्दे'],
  gu:['મુખ્ય તારણો','મહત્વપૂર્ણ તથ્યો','જોખમો','નિર્ણયો','ભલામણો','કાર્ય મુદ્દાઓ'],
  pa:['ਮੁੱਖ ਨਤੀਜੇ','ਮਹੱਤਵਪੂਰਨ ਤੱਥ','ਜੋਖਮ','ਫੈਸਲੇ','ਸਿਫ਼ਾਰਸ਼ਾਂ','ਕਾਰਵਾਈ ਦੇ ਨੁਕਤੇ'],
  ur:['اہم نتائج','اہم حقائق','خطرات','فیصلے','سفارشات','اقدامی نکات'],
 }
 const defaults=sectionNames.en
 const labels=sectionNames[r.language]||defaults
 const sections:[string,string[]|undefined][]=[...labels.map((label,i)=>[label,[r.findings,r.facts,r.risks,r.decisions,r.recommendations,r.action_items][i]] as [string,string[]|undefined])]
 return <div className="summary-sections">{sections.filter(([,items])=>items?.length).map(([label,items])=><section key={label}><h4>{label}</h4><ul>{items?.map((item,i)=><li key={i}>{item}</li>)}</ul></section>)}</div>
}

function EvaluationPanel({evaluation}:{evaluation:any}){
 if(!evaluation?.metrics)return null
 const labels:Record<string,string>={rouge1:'ROUGE-1',rouge2:'ROUGE-2',rougeL:'ROUGE-L',semanticSimilarity:'Semantic similarity',completeness:'Reference coverage',groundedness:'Groundedness',citationCoverage:'Citation coverage',relevance:'Relevance',factualConsistency:'Factual consistency',validationAccuracy:'Validation accuracy'}
 return <section className="evaluation-panel"><h4>Evaluation · individual metrics</h4><div className="evaluation-grid">{Object.entries(evaluation.metrics).map(([key,value]:[string,any])=>{const metricValue=typeof value==='number'?value:value?.value;const display=key.startsWith('rouge')?value?.f1:metricValue;return <div className="evaluation-metric" key={key}><small>{labels[key]||key}</small><b>{display==null?'Unavailable':typeof display==='number'?`${Math.round(display*100)}%`:String(display)}</b>{value?.reason&&<span>{value.reason}</span>}</div>})}</div><small>Completeness is ROUGE-L reference recall; semantic similarity and label-based metrics stay unavailable without their providers or gold labels.</small></section>
}

/** Wraps AIProcessingPipeline with the app's real state, never fake/random node states. */
function PipelineVisualization({resultStatus=null,regenerationCount=0,hasDoc=true,busy=false,error=false,audioActive=false}:{resultStatus?:string|null;regenerationCount?:number;hasDoc?:boolean;busy?:boolean;error?:boolean;audioActive?:boolean}){
 const steps=derivePipelineSteps({hasDoc,busy,error,resultStatus,regenerationCount,audioActive})
 return <AIProcessingPipeline steps={steps} compact/>
}

function DepthComparison({docId}:{docId:string}){
 const [data,setData]=useState<any>(null),[busy,setBusy]=useState(false),[error,setError]=useState('')
 async function compare(){setBusy(true);setError('');try{const r=await fetch(`${API}/evaluation/${docId}/compare-depths`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({})});const body=await r.json();if(!r.ok)throw Error(body.detail||'Depth comparison failed');setData(body.comparisons)}catch(e:any){setError(e.message)}finally{setBusy(false)}}
 return <section className="depth-comparison"><button className="secondary-btn" disabled={busy} onClick={compare}>{busy?'Running 3 analyses…':'Compare Quick / Standard / Deep'}</button>{error&&<p className="form-message">{error}</p>}{data&&<div className="evaluation-grid">{data.map((row:any)=><article className="evaluation-metric" key={row.depth}><b>{row.depth}</b><small>{row.summary.words} words · {row.processing_time_ms} ms · {row.summary.validated_claims} claims</small><span>ROUGE-L {row.metrics.rougeL.f1} · groundedness {row.metrics.groundedness.value??'Unavailable'} · citations {row.metrics.citationCoverage.value??'Unavailable'}</span></article>)}</div>}</section>
}

function ResultCard({result:r,claimCounts,setPage,docId,speechSegments=[],activeSegmentIndex=null,isAudioActive=false,audioLanguage=''}:{result:Result;claimCounts:Record<string,number>;setPage:(s:string)=>void;docId:string;speechSegments?:SpeechSegment[];activeSegmentIndex?:number|null;isAudioActive?:boolean;audioLanguage?:string}){
 const execSummary = r.executive_summary
 const executiveTitle:string=({en:'EXECUTIVE SUMMARY',hi:'कार्यकारी सारांश',te:'కార్యనిర్వాహక సారాంశం',kn:'ಕಾರ್ಯನಿರ್ವಾಹಕ ಸಾರಾಂಶ',ta:'செயற்குறிப்பு',ml:'എക്സിക്യൂട്ടീവ് സംഗ്രഹം',bn:'নির্বাহী সারাংশ',mr:'कार्यकारी सारांश',gu:'કાર્યકારી સારાંશ',pa:'ਕਾਰਜਕਾਰੀ ਸਾਰ',ur:'انتظامی خلاصہ'}[r.language]||'EXECUTIVE SUMMARY')
 const isAudioMatchingDoc = isAudioActive && (audioLanguage === r.language)
 const attempts=r.validation?.attempts??r.stats.validation_attempts??0
 const validationLabel=r.validation_status==='no_claims'?'NO CLAIMS TO VALIDATE':r.validation_provider==='local_basic'?'LOCAL BASIC CHECK':r.validation_status==='passed'||r.status==='validated'?'SUMMARY VALIDATED':r.validation_status==='unavailable'?'SUMMARY READY - VALIDATION INCOMPLETE':r.validation_status==='failed'||r.validation_status==='partial'||r.status==='partial'?'SUMMARY READY - PARTIAL VALIDATION':'SUMMARY READY'
 const validationNote=r.validation_status==='no_claims'?'No factual claims requiring validation were detected.':r.validation_provider==='local_basic'?'Validation service unavailable - basic local validation used; it is not equivalent to semantic validation.':r.validation_status==='unavailable'?`Validation service was unavailable after ${attempts} attempt${attempts===1?'':'s'}. The latest usable summary is shown without complete claim validation.`:r.validation_status==='partial'||r.validation_status==='failed'?`Summary generated with partial validation after ${attempts} attempt${attempts===1?'':'s'}. Some claims could not be fully verified against the source document. The latest usable summary is shown.`:attempts>1?`Validated after ${attempts} attempts.`:''
 const providerLabel=r.provider?.model?`${r.provider.name} · ${r.provider.model}`:r.provider?.name||'Local extractive fallback'
 return <div className="result-card"><div className="result-head"><div><span className="hero-kicker"><ShieldCheck size={13}/> {validationLabel}</span><h3>Analysis results</h3><p>{r.summary_depth_resolved||r.depth} summary{r.summary_depth_requested==='auto'?' (Auto requested)':''} · {r.purpose} · {r.language.toUpperCase()} · {providerLabel}</p><small>{validationNote}{r.provider?.fallback_used&&<><br/>External provider unavailable — generated using {r.provider.name}.</>}</small><small>{r.translation_validation_status==='partially_validated_against_original_report'?'Some translated claims did not pass the check against the original report.':''}{r.translation_validation_status==='translation_unavailable_source_fallback'||r.translation_validation_status==='translation_validation_failed_source_fallback'?'Translation was unavailable — showing the original-language summary.':''}{r.translation_provider&&r.provider&&r.translation_provider!==r.provider.name&&<>Translation fallback used {r.translation_provider}.</>}</small></div><div className="score-ring"><b>{r.quality.groundedness==null?'—':`${Math.round(r.quality.groundedness*100)}%`}</b><small>source match</small></div></div><div className="summary-copy"><div style={{display:'flex',justifyContent:'space-between',alignItems:'center',marginBottom:'0.5rem'}}><label className="field-label" style={{margin:0}}>{executiveTitle}</label>{isAudioActive&&(isAudioMatchingDoc&&activeSegmentIndex!==null?<span className="speech-segment-badge"><Volume2 size={12}/> Live Speech Sync</span>:<span className="speech-segment-badge" style={{opacity:0.8}}><Volume2 size={12}/> Audio: {languageNames[audioLanguage]||audioLanguage.toUpperCase()}</span>)}</div>{execSummary?<div className="audio-summary-transcript">{speechSegments.length>0&&isAudioMatchingDoc?speechSegments.map((seg,i)=><span key={i} className={`speech-segment ${activeSegmentIndex===i?'speech-segment--active':''}`}>{seg.text}{' '}</span>):<p>{execSummary}</p>}</div>:<p className="muted-text">{r.source_language&&r.language!==r.source_language?`Executive summary unavailable in ${r.language.toUpperCase()} — cross-lingual validation returned no supported claims. The source-language summary is shown above in the claims section.`:'No executive summary was generated for this document.'}</p>}</div><PipelineVisualization hasDoc resultStatus={r.status} regenerationCount={r.stats.regeneration_count||0}/>{!!r.stats.regeneration_count&&(r.validation_status==='passed'||r.status==='validated')&&<p className="regeneration-note"><Sparkles size={13}/> The validation agent found unsupported claims and regenerated the summary {r.stats.regeneration_count} time{r.stats.regeneration_count===1?'':'s'} before this version passed validation.</p>}<SummarySections result={r}/><div className="result-stats"><span><b>{r.stats.words.toLocaleString()}</b> source words</span><span><b>{r.stats.context_chunks_used??r.stats.chunks}</b> source chunks used</span><span><b>{r.quality.claims_checked}</b> claims checked</span><span><b>{claimCounts.SUPPORTED||0}</b> supported</span>{r.stats.processing_time_ms!=null&&<span><b>{r.stats.processing_time_ms} ms</b> processing</span>}</div><EvaluationPanel evaluation={r.evaluation}/>{r.evaluation?.metrics&&<DepthComparison docId={docId}/>}<div className="result-actions"><button className="secondary-btn" onClick={()=>setPage('Sources')}><BookOpen size={15}/> Explore sources</button><button className="secondary-btn" onClick={()=>setPage('Ask document')}><MessageSquareText size={15}/> Ask document</button><button className="secondary-btn" onClick={()=>setPage('Audio briefing')}><Headphones size={15}/> Audio briefing</button></div><details className="claim-details"><summary>Claim validation · {r.claims.length} claims <ChevronDown size={15}/></summary>{r.claims.map(c=><div className="claim-row" key={c.id}><span className={`claim-check ${c.status.toLowerCase()}`}>{c.status==='SUPPORTED'?<Check size={14}/>:c.status==='UNSUPPORTED'?<X size={14}/>:<CircleHelp size={14}/>}</span><div><b>{c.claim}</b><small>{c.status} · confidence {c.confidence==null?'unavailable':`${Math.round(c.confidence*100)}%`} · {c.reason} · {c.sources[0]?.chunk_id||'No source'}</small>{c.evidence&&<p>{c.evidence}</p>}</div></div>)}</details>{r.limitations.map((l,i)=><div className="limitation" key={i}><span>i</span>{l}</div>)}</div>
}
function Sources({doc}:{doc:Doc}){const [sources,setSources]=useState<any[]>([]);useEffect(()=>{fetch(`${API}/documents/${doc.id}/sources`).then(r=>r.json()).then(setSources)},[doc]);return <div className="source-list">{sources.map((s,index)=><article key={s.id||index} className="source-card"><div className="source-meta"><span>{s.section}</span><span>{s.page?`Page ${s.page}`:s.id}</span><span>{s.language.toUpperCase()}</span></div><p>{s.text}</p><small><span className="green-dot"/> Extracted source passage · {s.id}</small></article>)}</div>}
function Compare(){const [a,setA]=useState<File|null>(null),[b,setB]=useState<File|null>(null),[busy,setBusy]=useState(false),[data,setData]=useState<any>(null);async function run(){if(!a||!b)return;setBusy(true);const fd=new FormData();fd.append('file_a',a);fd.append('file_b',b);const r=await fetch(`${API}/documents/compare`,{method:'POST',body:fd});setData(await r.json());setBusy(false)}return <><PageHeading title="Compare versions" subtitle="See exact sentence additions and removals between document versions."/><div className="compare-card"><div className="compare-inputs">{[[a,setA,'Version A'],[b,setB,'Version B']].map(([f,set,label]:any)=><label key={label} className="compare-drop"><Upload/><b>{f?.name||`Add ${label}`}</b><small>PDF, DOCX, TXT or JSON · 25 MB max</small><input type="file" accept=".pdf,.docx,.txt,.json" onChange={e=>set(e.target.files?.[0]||null)}/></label>)}</div><button className="primary-btn" disabled={!a||!b||busy} onClick={run}>{busy?'Comparing…':'Compare documents →'}</button>{data&&<><CompareVisualization added={data.added} removed={data.removed}/><div className="diff-grid"><div><h3 className="added">+ Added ({data.added.length})</h3>{data.added.map((x:string,i:number)=><p key={`${i}-${x}`}>{x}</p>)}</div><div><h3 className="removed">− Removed ({data.removed.length})</h3>{data.removed.map((x:string,i:number)=><p key={`${i}-${x}`}>{x}</p>)}</div><small>{data.method}. {data.limitations}</small></div></>}</div></>}
