<script setup>
import { computed, nextTick, onMounted, ref } from 'vue'
import { marked } from 'marked'
import DOMPurify from 'dompurify'
import { askQuestion, fetchInsights, uploadPdf } from './lib/api'

marked.setOptions({
  gfm: true,
  breaks: true
})

function renderMarkdown(content) {
  if (!content) return ''
  return DOMPurify.sanitize(marked.parse(String(content)))
}

const activeTab = ref('chat')
const selectedFile = ref(null)
const fileInput = ref(null)
const isDragging = ref(false)
const chatPanel = ref(null)
const uploadState = ref({ loading: false, error: '', success: '' })
const askState = ref({ loading: false, error: '' })
const documentId = ref('')
const pdfName = ref('')
const chunkCount = ref(0)
const question = ref('')
const chatMessages = ref([
  {
    role: 'assistant',
    content: 'Welcome to SmartDoc. Upload a PDF and I’ll help you find clear, grounded answers from it.',
    sources: []
  }
])
const insights = ref({
  total_questions: 0,
  most_asked_questions: [],
  questions_per_day: [],
  latest_pdf_name: null
})

const hasDocument = computed(() => Boolean(documentId.value))
const fileSize = computed(() => {
  if (!selectedFile.value) return ''
  const size = selectedFile.value.size / 1024 / 1024
  return `${size < 1 ? (size * 1024).toFixed(0) + ' KB' : size.toFixed(1) + ' MB'}`
})
const documentLabel = computed(() => pdfName.value || 'No document loaded')
const progressLabel = computed(() => uploadState.value.loading ? 'Reading your document…' : 'Ready to analyze')

const suggestedQuestions = [
  'Give me a concise summary',
  'What are the key recommendations?',
  'Which dates or deadlines matter?'
]

function onFileChange(event) {
  const [file] = event.target.files || []
  setSelectedFile(file)
}

function setSelectedFile(file) {
  if (!file) return
  if (file.type !== 'application/pdf' && !file.name.toLowerCase().endsWith('.pdf')) {
    uploadState.value = { loading: false, error: 'Please choose a PDF file.', success: '' }
    return
  }
  selectedFile.value = file
  uploadState.value = { loading: false, error: '', success: '' }
}

function onDrop(event) {
  isDragging.value = false
  const [file] = event.dataTransfer.files || []
  setSelectedFile(file)
}

function openFilePicker() {
  fileInput.value?.click()
}

async function handleUpload() {
  if (!selectedFile.value) {
    uploadState.value = { loading: false, error: 'Choose a PDF before uploading.', success: '' }
    return
  }

  uploadState.value = { loading: true, error: '', success: '' }
  try {
    const response = await uploadPdf(selectedFile.value)
    documentId.value = response.document_id
    pdfName.value = response.pdf_name
    chunkCount.value = response.chunk_count
    chatMessages.value = [{
      role: 'assistant',
      content: `${response.pdf_name} is ready. Ask me anything about the document and I’ll cite the relevant passages.`,
      sources: []
    }]
    uploadState.value = {
      loading: false,
      error: '',
      success: 'Document indexed and ready for questions.'
    }
    activeTab.value = 'chat'
    await nextTick()
    chatPanel.value?.scrollTo({ top: chatPanel.value.scrollHeight, behavior: 'smooth' })
  } catch (error) {
    uploadState.value = { loading: false, error: error.message, success: '' }
  }
}

async function handleAsk(prompt = question.value) {
  const trimmed = prompt.trim()
  if (!trimmed || !documentId.value || askState.value.loading) return

  askState.value = { loading: true, error: '' }
  chatMessages.value.push({ role: 'user', content: trimmed, sources: [] })
  question.value = ''
  await nextTick()
  chatPanel.value?.scrollTo({ top: chatPanel.value.scrollHeight, behavior: 'smooth' })

  try {
    const response = await askQuestion({ document_id: documentId.value, question: trimmed })
    chatMessages.value.push({ role: 'assistant', content: response.answer, sources: response.sources || [] })
    await loadInsights()
  } catch (error) {
    askState.value = { loading: false, error: error.message }
    chatMessages.value.push({ role: 'assistant', content: `I couldn’t complete that request. ${error.message}`, sources: [] })
  }

  askState.value = { loading: false, error: '' }
  await nextTick()
  chatPanel.value?.scrollTo({ top: chatPanel.value.scrollHeight, behavior: 'smooth' })
}

async function loadInsights() {
  try {
    insights.value = await fetchInsights()
  } catch (error) {
    insights.value = { total_questions: 0, most_asked_questions: [], questions_per_day: [], latest_pdf_name: null, error: error.message }
  }
}

function formatDay(day) {
  return new Date(day).toLocaleDateString(undefined, { month: 'short', day: 'numeric' })
}

function formatScore(score) {
  return Math.round((score || 0) * 100)
}

onMounted(loadInsights)
</script>

<template>
  <div class="app-shell">
    <header class="topbar">
      <div class="brand-lockup">
        <div class="brand-mark"><span></span><span></span><span></span></div>
        <div><strong>SmartDoc</strong><small>AI document intelligence</small></div>
      </div>
      <div class="topbar-actions">
        <span class="secure-pill"><i></i> Private workspace</span>
        <button class="avatar" aria-label="Account">AB</button>
      </div>
    </header>

    <main class="workspace">
      <section class="welcome-row">
        <div>
          <p class="eyebrow">DOCUMENT WORKSPACE</p>
          <h1>Make sense of your documents.</h1>
          <p class="lede">Upload a PDF, then ask focused questions with answers grounded in the source.</p>
        </div>
        <div class="system-status"><span class="status-dot"></span><span>AI systems operational</span></div>
      </section>

      <section class="workspace-grid">
        <aside class="sidebar">
          <div class="sidebar-card upload-card">
            <div class="card-heading">
              <div class="heading-icon blue-icon">↑</div>
              <div><h2>Add a document</h2><p>PDF files up to 20 MB</p></div>
            </div>
            <button
              class="drop-zone"
              :class="{ dragging: isDragging, 'has-file': selectedFile }"
              @click="openFilePicker"
              @dragover.prevent="isDragging = true"
              @dragleave.prevent="isDragging = false"
              @drop.prevent="onDrop"
            >
              <input ref="fileInput" type="file" accept="application/pdf" hidden @change="onFileChange" />
              <span class="upload-cloud">↥</span>
              <strong>{{ selectedFile ? selectedFile.name : 'Drop your PDF here' }}</strong>
              <span>{{ selectedFile ? fileSize + ' · Ready to upload' : 'or click to browse files' }}</span>
            </button>
            <button class="primary-button full-width" :disabled="uploadState.loading || !selectedFile" @click="handleUpload">
              <span v-if="uploadState.loading" class="spinner"></span>
              {{ uploadState.loading ? 'Indexing document…' : 'Upload and analyze' }}
            </button>
            <p v-if="uploadState.success" class="feedback success">✓ {{ uploadState.success }}</p>
            <p v-if="uploadState.error" class="feedback error">{{ uploadState.error }}</p>
          </div>

          <div class="sidebar-card current-card">
            <div class="card-heading compact">
              <div class="heading-icon purple-icon">▤</div>
              <div><h2>Current document</h2><p>{{ hasDocument ? 'Ready to explore' : 'Nothing uploaded yet' }}</p></div>
            </div>
            <div class="document-preview" :class="{ empty: !hasDocument }">
              <div class="pdf-badge">PDF</div>
              <div class="doc-details"><strong>{{ documentLabel }}</strong><span v-if="hasDocument">{{ chunkCount }} searchable sections</span><span v-else>Upload a file to get started</span></div>
              <span v-if="hasDocument" class="check-badge">✓</span>
            </div>
          </div>

          <div class="sidebar-card tip-card">
            <span class="tip-icon">✦</span>
            <div><strong>Get better answers</strong><p>Ask specific questions and SmartDoc will surface the most relevant passages.</p></div>
          </div>
        </aside>

        <section class="content-area">
          <nav class="tabs" aria-label="Workspace sections">
            <button :class="{ active: activeTab === 'chat' }" @click="activeTab = 'chat'"><span>✧</span> Ask your document</button>
            <button :class="{ active: activeTab === 'insights' }" @click="activeTab = 'insights'; loadInsights()"><span>▥</span> Insights <b v-if="insights.total_questions">{{ insights.total_questions }}</b></button>
          </nav>

          <section v-if="activeTab === 'chat'" class="chat-card">
            <div class="chat-header">
              <div><p class="eyebrow">CONVERSATION</p><h2>Ask your document</h2></div>
              <div class="grounded-pill"><span class="status-dot"></span> Grounded answers</div>
            </div>
            <div ref="chatPanel" class="messages">
              <div v-if="!hasDocument" class="empty-state">
                <div class="empty-orb">✦</div>
                <h3>Your document, made searchable.</h3>
                <p>Upload a PDF from the left to unlock summaries, answers, and source-backed insights.</p>
              </div>
              <article v-for="(message, index) in chatMessages" :key="index" class="message" :class="message.role">
                <div class="message-avatar" :class="message.role === 'assistant' ? 'ai-avatar' : 'user-avatar'">{{ message.role === 'assistant' ? '✦' : 'AB' }}</div>
                <div class="message-body">
                  <div class="message-meta"><strong>{{ message.role === 'assistant' ? 'SmartDoc AI' : 'You' }}</strong><span>{{ message.role === 'assistant' ? 'Source-grounded response' : 'Question' }}</span></div>
                  <div class="formatted-content" v-html="renderMarkdown(message.content)"></div>
                  <div v-if="message.sources?.length" class="sources">
                    <div class="sources-label">Relevant passages <span>{{ message.sources.length }}</span></div>
                    <details v-for="source in message.sources" :key="`${index}-${source.rank}`" class="source-item">
                      <summary><span>Source {{ source.rank }}</span><em>{{ formatScore(source.score) }}% match</em></summary>
                      <p>{{ source.content }}</p>
                    </details>
                  </div>
                </div>
              </article>
              <article v-if="askState.loading" class="message assistant">
                <div class="message-avatar ai-avatar">✦</div>
                <div class="message-body"><div class="message-meta"><strong>SmartDoc AI</strong><span>Thinking</span></div><div class="typing"><i></i><i></i><i></i></div></div>
              </article>
            </div>
            <div class="composer-wrap">
              <div v-if="!hasDocument" class="composer-hint">Upload a document to start asking questions</div>
              <div v-else class="suggestions"><button v-for="suggestion in suggestedQuestions" :key="suggestion" @click="handleAsk(suggestion)">{{ suggestion }}</button></div>
              <div class="composer">
                <input v-model="question" :disabled="!hasDocument || askState.loading" placeholder="Ask anything about your document…" @keyup.enter="handleAsk()" />
                <button class="send-button" aria-label="Send question" :disabled="!hasDocument || askState.loading || !question.trim()" @click="handleAsk()">↑</button>
              </div>
              <p class="composer-note">SmartDoc only answers from your uploaded document · Press Enter to send</p>
              <p v-if="askState.error" class="feedback error">{{ askState.error }}</p>
            </div>
          </section>

          <section v-else class="insights-card">
            <div class="chat-header"><div><p class="eyebrow">WORKSPACE ANALYTICS</p><h2>Insights at a glance</h2></div><button class="ghost-button" @click="loadInsights">↻ Refresh</button></div>
            <div class="metric-grid">
              <div class="metric"><span class="metric-icon blue-icon">◎</span><div><span>Total questions</span><strong>{{ insights.total_questions }}</strong></div></div>
              <div class="metric"><span class="metric-icon purple-icon">▤</span><div><span>Latest document</span><strong class="truncate">{{ insights.latest_pdf_name || 'No activity yet' }}</strong></div></div>
              <div class="metric"><span class="metric-icon green-icon">✦</span><div><span>Top questions</span><strong>{{ insights.most_asked_questions.length }}</strong></div></div>
            </div>
            <div class="insights-columns">
              <div class="insight-panel"><div class="panel-title"><h3>Most asked questions</h3><span>Ranked by frequency</span></div><div v-if="insights.most_asked_questions.length" class="question-list"><div v-for="item in insights.most_asked_questions" :key="item.question"><span class="rank">#</span><p>{{ item.question }}</p><b>{{ item.count }}</b></div></div><div v-else class="no-data">Your most common questions will appear here.</div></div>
              <div class="insight-panel"><div class="panel-title"><h3>Questions per day</h3><span>Recent activity</span></div><div v-if="insights.questions_per_day.length" class="day-list"><div v-for="item in insights.questions_per_day" :key="item.day"><span>{{ formatDay(item.day) }}</span><div class="bar"><i :style="{ width: `${Math.min(100, item.count * 15 + 10)}%` }"></i></div><b>{{ item.count }}</b></div></div><div v-else class="no-data">Daily activity will appear once you start chatting.</div></div>
            </div>
            <p v-if="insights.error" class="feedback error">{{ insights.error }}</p>
          </section>
        </section>
      </section>
    </main>
    <footer><span>SmartDoc AI</span><span>Private by design · Answers stay grounded in your source</span></footer>
  </div>
</template>
