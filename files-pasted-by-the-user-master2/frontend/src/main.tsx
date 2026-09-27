import React, { Component, type ErrorInfo, type ReactNode } from 'react'
import { createRoot } from 'react-dom/client'
import App from './App'
import './style.css'

class StartupBoundary extends Component<{children:ReactNode},{error:string|null}>{
  state={error:null as string|null}
  static getDerivedStateFromError(error:Error){return {error:error.message||'Unknown rendering error'}}
  componentDidCatch(error:Error,info:ErrorInfo){console.error('Scribe render failed',error,info.componentStack)}
  render(){
    if(this.state.error)return <main className="startup-screen startup-error" role="alert"><div className="startup-mark">!</div><h1>Scribe could not render</h1><p>The UI encountered a startup error. Check the browser console and Vite terminal, then reload after correcting it.</p><pre>{this.state.error}</pre></main>
    return this.props.children
  }
}

const root=document.getElementById('root')
if(!root)throw new Error('Missing #root mount element')
createRoot(root).render(<React.StrictMode><StartupBoundary><App /></StartupBoundary></React.StrictMode>)
