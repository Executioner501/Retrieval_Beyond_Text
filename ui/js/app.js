/**
 * Context-Aware Conversational RAG Studio - Interactive Simulation Engine
 * Authors: Team B10 (Sai Dinesh, Satya Sri Dheeraj, Sai Nikhil)
 */

document.addEventListener('DOMContentLoaded', () => {
  // State variables
  let filterSystemNotices = true;
  let alpha = 0.40;
  let lambda = 1.20;
  let topK = 3;
  let currentQuery = "What is our latest plan for the frontend?";

  // 1. Raw Parsed Messages Data (matches sample_chat.txt)
  const messagesData = [
    { id: "m1", time: "10/01/2026, 09:30", sender: "System", text: "Messages and calls are end-to-end encrypted. No one outside of this chat, not even WhatsApp, can read or listen to them.", is_system: true, topic: "System" },
    { id: "m2", time: "10/01/2026, 09:31", sender: "System", text: "Rahul added Sai and Dheeraj", is_system: true, topic: "System" },
    { id: "m3", time: "10/01/2026, 09:32", sender: "Rahul", text: "Hey guys, starting this group for our final year project discussions.", is_system: false, topic: "Tech Stack" },
    { id: "m4", time: "10/01/2026, 09:33", sender: "Sai", text: "Awesome! Let's decide on the tech stack first.", is_system: false, topic: "Tech Stack" },
    { id: "m5", time: "10/01/2026, 09:34", sender: "Dheeraj", text: "Agreed. Are we building the frontend in React or Flutter?", is_system: false, topic: "Tech Stack" },
    { id: "m6", time: "10/01/2026, 09:35", sender: "Rahul", text: "For now, let's finalize React for the web dashboard.", is_system: false, topic: "Tech Stack" },
    { id: "m7", time: "10/01/2026, 09:36", sender: "Sai", text: "Sounds good. React it is.", is_system: false, topic: "Tech Stack" },
    { id: "m8", time: "10/01/2026, 09:38", sender: "Rahul", text: "<Media omitted>", is_system: false, has_media: true, topic: "Architecture" },
    { id: "m9", time: "10/01/2026, 09:39", sender: "Rahul", text: "Check out this architecture reference diagram. Let me know what you think about the retrieval flow.", is_system: false, topic: "Architecture" },
    { id: "m10", time: "10/01/2026, 09:40", sender: "Dheeraj", text: "Looks solid.", is_system: false, topic: "Architecture" },

    { id: "m11", time: "10/01/2026, 12:45", sender: "Sai", text: "Anyone hungry? Let's grab lunch at the canteen.", is_system: false, topic: "Lunch" },
    { id: "m12", time: "10/01/2026, 12:46", sender: "Dheeraj", text: "On my way!", is_system: false, topic: "Lunch" },
    { id: "m13", time: "10/01/2026, 12:47", sender: "Rahul", text: "Same, 5 mins.", is_system: false, topic: "Lunch" },

    { id: "m14", time: "12/08/2026, 10:30", sender: "Rahul", text: "Are we meeting tomorrow for the guide review?", is_system: false, topic: "Guide Review" },
    { id: "m15", time: "12/08/2026, 10:31", sender: "Sai", text: "Yes, after class around 4 PM.", is_system: false, topic: "Guide Review" },
    { id: "m16", time: "12/08/2026, 10:31", sender: "Dheeraj", text: "Who was bringing the documents and proposal printouts?", is_system: false, topic: "Guide Review" },
    { id: "m17", time: "12/08/2026, 10:32", sender: "Rahul", text: "I will bring the documents tomorrow.", is_system: false, topic: "Guide Review" },
    { id: "m18", time: "12/08/2026, 10:33", sender: "Sai", text: "Okay.", is_system: false, topic: "Guide Review" },
    { id: "m19", time: "12/08/2026, 10:33", sender: "Dheeraj", text: "Great.", is_system: false, topic: "Guide Review" },

    { id: "m20", time: "12/08/2026, 10:35", sender: "Sai", text: "By the way, where should we go for the semester break trip?", is_system: false, topic: "Goa Vacation" },
    { id: "m21", time: "12/08/2026, 10:36", sender: "Dheeraj", text: "Goa sounds good.", is_system: false, topic: "Goa Vacation" },
    { id: "m22", time: "12/08/2026, 10:36", sender: "Rahul", text: "Let's finalize Goa for December.", is_system: false, topic: "Goa Vacation" },
    { id: "m23", time: "12/08/2026, 10:37", sender: "Sai", text: "Nice, we should book train tickets next month.", is_system: false, topic: "Goa Vacation" },

    { id: "m24", time: "12/08/2026, 10:38", sender: "Rahul", text: "Did you finish the machine learning assignment submission due tonight?", is_system: false, topic: "Assignment" },
    { id: "m25", time: "12/08/2026, 10:39", sender: "Sai", text: "Not yet, working on question 3.", is_system: false, topic: "Assignment" },
    { id: "m26", time: "12/08/2026, 10:40", sender: "Dheeraj", text: "Same here.", is_system: false, topic: "Assignment" },

    { id: "m27", time: "15/08/2026, 14:00", sender: "Rahul", text: "Urgent update regarding the project implementation.", is_system: false, topic: "Tech Stack Revision" },
    { id: "m28", time: "15/08/2026, 14:02", sender: "Rahul", text: "Bindhya ma'am suggested our prototype should be available on mobile devices for testing.", is_system: false, topic: "Tech Stack Revision" },
    { id: "m29", time: "15/08/2026, 14:03", sender: "Dheeraj", text: "So what does that mean for our frontend?", is_system: false, topic: "Tech Stack Revision" },
    { id: "m30", time: "15/08/2026, 14:05", sender: "Rahul", text: "We changed the plan and will use Flutter instead of React so we can run on Android and iOS.", is_system: false, topic: "Tech Stack Revision" },
    { id: "m31", time: "15/08/2026, 14:06", sender: "Sai", text: "Understood! I'll update the proposal documentation with Flutter.", is_system: false, topic: "Tech Stack Revision" },
    { id: "m32", time: "15/08/2026, 14:07", sender: "Dheeraj", text: "Perfect.", is_system: false, topic: "Tech Stack Revision" }
  ];

  // 2. Chunking Data Structures
  const naiveChunks = [
    {
      id: "NAIVE-CHUNK-01",
      title: "Session Jan 10 (Morning)",
      timeRange: "10/01/2026, 09:32 - 09:40",
      msgsCount: 8,
      text: "Rahul: Hey guys, starting this group for our final year project discussions.\nSai: Awesome! Let's decide on the tech stack first.\nDheeraj: Agreed. Are we building the frontend in React or Flutter?\nRahul: For now, let's finalize React for the web dashboard.\nSai: Sounds good. React it is.\nRahul: Check out this architecture reference diagram...\nDheeraj: Looks solid.",
      issue: "Acceptable turn density, but merged architecture review with initial frontend decision."
    },
    {
      id: "NAIVE-CHUNK-02",
      title: "Session Jan 10 (Lunch)",
      timeRange: "10/01/2026, 12:45 - 12:47",
      msgsCount: 3,
      text: "Sai: Anyone hungry? Let's grab lunch at the canteen.\nDheeraj: On my way!\nRahul: Same, 5 mins.",
      issue: "Colloquial chatter chunk with zero factual task knowledge."
    },
    {
      id: "NAIVE-CHUNK-03",
      title: "Session Aug 12 (Mega-Blob)",
      timeRange: "12/08/2026, 10:30 - 10:40",
      msgsCount: 13,
      text: "Rahul: Are we meeting tomorrow for the guide review?\nSai: Yes, after class around 4 PM.\nDheeraj: Who was bringing the documents and proposal printouts?\nRahul: I will bring the documents tomorrow.\nSai: Okay.\nDheeraj: Great.\nSai: By the way, where should we go for the semester break trip?\nDheeraj: Goa sounds good.\nRahul: Let's finalize Goa for December.\nSai: Nice, we should book train tickets next month.\nRahul: Did you finish the machine learning assignment submission due tonight?\nSai: Not yet, working on question 3.\nDheeraj: Same here.",
      issue: "CRITICAL FAILURE: 3 completely unrelated topics (guide review documents, vacation to Goa, ML homework) merged into 1 chunk because messages occurred within 10 minutes!"
    },
    {
      id: "NAIVE-CHUNK-04",
      title: "Session Aug 15 (Update)",
      timeRange: "15/08/2026, 14:00 - 14:07",
      msgsCount: 6,
      text: "Rahul: Urgent update regarding the project implementation.\nRahul: Bindhya ma'am suggested our prototype should be available on mobile devices for testing.\nDheeraj: So what does that mean for our frontend?\nRahul: We changed the plan and will use Flutter instead of React so we can run on Android and iOS.\nSai: Understood! I'll update the proposal documentation with Flutter.\nDheeraj: Perfect.",
      issue: "New decision exists, but naive similarity search often prefers older Chunk-01 due to word overlap."
    }
  ];

  const contextChunks = [
    {
      id: "CTX-CHUNK-01",
      topic: "Tech Stack • Initial Selection",
      time: "2026-01-10T09:32",
      timeDisplay: "10/01/2026, 09:32 - 09:36",
      participants: ["Rahul", "Sai", "Dheeraj"],
      attributedText: "[10/01/2026 09:34] Dheeraj: Are we building frontend in React or Flutter?\n[10/01/2026 09:35] Rahul: For now, let's finalize React for the web dashboard.\n[10/01/2026 09:36] Sai: Sounds good. React it is.",
      daysAgo: 217,
      baseSim: {
        "frontend": 0.88,
        "documents": 0.12,
        "trip": 0.05,
        "assignment": 0.08
      }
    },
    {
      id: "CTX-CHUNK-02",
      topic: "System Architecture Review",
      time: "2026-01-10T09:38",
      timeDisplay: "10/01/2026, 09:38 - 09:40",
      participants: ["Rahul", "Dheeraj"],
      attributedText: "[10/01/2026 09:39] Rahul: Check out this architecture reference diagram. Let me know what you think about the retrieval flow.\n[10/01/2026 09:40] Dheeraj: Looks solid.",
      daysAgo: 217,
      baseSim: {
        "frontend": 0.35,
        "documents": 0.20,
        "trip": 0.02,
        "assignment": 0.10
      }
    },
    {
      id: "CTX-CHUNK-03",
      topic: "Guide Review & Documents Commitment",
      time: "2026-08-12T10:30",
      timeDisplay: "12/08/2026, 10:30 - 10:33",
      participants: ["Rahul", "Sai", "Dheeraj"],
      attributedText: "[12/08/2026 10:30] Rahul: Are we meeting tomorrow for guide review?\n[12/08/2026 10:31] Sai: Yes, after class around 4 PM.\n[12/08/2026 10:31] Dheeraj: Who was bringing the documents and proposal printouts?\n[12/08/2026 10:32] Rahul: I will bring the documents tomorrow.\n[12/08/2026 10:33] Sai: Okay.\n[12/08/2026 10:33] Dheeraj: Great.",
      daysAgo: 3,
      baseSim: {
        "frontend": 0.15,
        "documents": 0.94,
        "trip": 0.08,
        "assignment": 0.22
      }
    },
    {
      id: "CTX-CHUNK-04",
      topic: "Semester Vacation • Goa Decision",
      time: "2026-08-12T10:35",
      timeDisplay: "12/08/2026, 10:35 - 10:37",
      participants: ["Sai", "Dheeraj", "Rahul"],
      attributedText: "[12/08/2026 10:35] Sai: By the way, where should we go for the semester break trip?\n[12/08/2026 10:36] Dheeraj: Goa sounds good.\n[12/08/2026 10:36] Rahul: Let's finalize Goa for December.\n[12/08/2026 10:37] Sai: Nice, we should book train tickets next month.",
      daysAgo: 3,
      baseSim: {
        "frontend": 0.05,
        "documents": 0.08,
        "trip": 0.92,
        "assignment": 0.04
      }
    },
    {
      id: "CTX-CHUNK-05",
      topic: "Coursework • ML Assignment",
      time: "2026-08-12T10:38",
      timeDisplay: "12/08/2026, 10:38 - 10:40",
      participants: ["Rahul", "Sai", "Dheeraj"],
      attributedText: "[12/08/2026 10:38] Rahul: Did you finish the machine learning assignment submission due tonight?\n[12/08/2026 10:39] Sai: Not yet, working on question 3.\n[12/08/2026 10:40] Dheeraj: Same here.",
      daysAgo: 3,
      baseSim: {
        "frontend": 0.12,
        "documents": 0.18,
        "trip": 0.03,
        "assignment": 0.89
      }
    },
    {
      id: "CTX-CHUNK-06",
      topic: "Architecture Shift • Flutter Adoption",
      time: "2026-08-15T14:00",
      timeDisplay: "15/08/2026, 14:00 - 14:07",
      participants: ["Rahul", "Dheeraj", "Sai"],
      attributedText: "[15/08/2026 14:02] Rahul: Bindhya ma'am suggested our prototype should be available on mobile devices.\n[15/08/2026 14:03] Dheeraj: So what does that mean for our frontend?\n[15/08/2026 14:05] Rahul: We changed the plan and will use Flutter instead of React so we can run on Android and iOS.\n[15/08/2026 14:06] Sai: Understood! I'll update proposal with Flutter.",
      daysAgo: 0,
      baseSim: {
        "frontend": 0.86,
        "documents": 0.10,
        "trip": 0.02,
        "assignment": 0.06
      }
    }
  ];

  // Tab switching logic
  const tabButtons = document.querySelectorAll('.tab-button');
  const tabContents = document.querySelectorAll('.tab-content');

  tabButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      tabButtons.forEach(b => b.classList.remove('active'));
      tabContents.forEach(c => c.classList.remove('active'));

      btn.classList.add('active');
      const targetId = btn.getAttribute('data-tab');
      document.getElementById(targetId)?.classList.add('active');
    });
  });

  // Render Chat Stream
  function renderChatStream() {
    const list = document.getElementById('chat-stream-list');
    if (!list) return;
    list.innerHTML = '';

    messagesData.forEach(msg => {
      if (filterSystemNotices && msg.is_system) return;

      if (msg.is_system) {
        const sysRow = document.createElement('div');
        sysRow.className = 'chat-system-row';
        sysRow.innerHTML = `<span>🔒 [${msg.time}] ${msg.text}</span>`;
        list.appendChild(sysRow);
        return;
      }

      const row = document.createElement('div');
      row.className = 'chat-message-row';

      const avatarClass = msg.sender.toLowerCase().includes('rahul') ? 'avatar-rahul' :
                          msg.sender.toLowerCase().includes('sai') ? 'avatar-sai' : 'avatar-dheeraj';
      const initial = msg.sender.charAt(0).toUpperCase();

      row.innerHTML = `
        <div class="chat-avatar ${avatarClass}">${initial}</div>
        <div class="chat-content-wrap">
          <div class="chat-meta">
            <span class="chat-sender">${msg.sender}</span>
            <span class="chat-time">${msg.time}</span>
            <span class="badge badge-cyan" style="font-size: 0.65rem; padding: 0.1rem 0.4rem;">${msg.topic}</span>
          </div>
          <div class="chat-text">${msg.has_media ? '<em>&lt;Media omitted&gt;</em>' : msg.text}</div>
        </div>
      `;
      list.appendChild(row);
    });
  }

  // Render Chunking comparison
  function renderChunkComparison() {
    const naiveBox = document.getElementById('naive-chunks-container');
    const ctxBox = document.getElementById('context-chunks-container');
    if (!naiveBox || !ctxBox) return;

    naiveBox.innerHTML = '';
    naiveChunks.forEach(c => {
      const card = document.createElement('div');
      card.className = 'chunk-card naive-fail';
      card.innerHTML = `
        <div class="chunk-header">
          <span class="chunk-id">${c.id} • ${c.msgsCount} msgs</span>
          <span class="badge badge-amber" style="font-size: 0.7rem;">${c.timeRange}</span>
        </div>
        <div class="chunk-body">${c.text}</div>
        <div style="font-size: 0.75rem; color: var(--accent-rose); margin-top: 0.5rem; font-weight: 500;">
          ⚠️ ${c.issue}
        </div>
      `;
      naiveBox.appendChild(card);
    });

    ctxBox.innerHTML = '';
    contextChunks.forEach(c => {
      const card = document.createElement('div');
      card.className = 'chunk-card context-success';
      card.innerHTML = `
        <div class="chunk-header">
          <span class="chunk-id">${c.id}</span>
          <span class="badge badge-cyan" style="font-size: 0.7rem;">${c.timeDisplay}</span>
        </div>
        <div class="topic-tag">${c.topic}</div>
        <div class="chunk-body" style="margin-top: 0.5rem;">${c.attributedText}</div>
        <div style="display: flex; gap: 0.5rem; margin-top: 0.5rem;">
          <span class="badge badge-violet" style="font-size: 0.68rem;">Speakers: ${c.participants.join(', ')}</span>
        </div>
      `;
      ctxBox.appendChild(card);
    });
  }

  // Calculate similarity and time-decay rank
  function calculateRankings() {
    const queryLower = currentQuery.toLowerCase();
    let queryKey = "frontend";
    if (queryLower.includes("document") || queryLower.includes("review")) queryKey = "documents";
    else if (queryLower.includes("trip") || queryLower.includes("vacation") || queryLower.includes("goa")) queryKey = "trip";
    else if (queryLower.includes("assignment") || queryLower.includes("machine learning")) queryKey = "assignment";

    const maxDays = 220;

    const scored = contextChunks.map(chunk => {
      const sim = chunk.baseSim[queryKey] || 0.1;
      const normalizedDeltaT = chunk.daysAgo / maxDays;
      const timeDecay = Math.exp(-lambda * normalizedDeltaT);
      const finalScore = (alpha * sim) + ((1 - alpha) * timeDecay);

      return {
        chunk,
        sim,
        timeDecay,
        finalScore
      };
    });

    // Sort descending by finalScore
    scored.sort((a, b) => b.finalScore - a.finalScore);

    renderRankedCandidates(scored.slice(0, topK));
    updateGroundedAnswer(queryKey, scored[0]);
  }

  function renderRankedCandidates(candidates) {
    const container = document.getElementById('ranked-candidates-container');
    if (!container) return;
    container.innerHTML = '';

    candidates.forEach((item, index) => {
      const card = document.createElement('div');
      card.className = 'ranked-item';
      card.innerHTML = `
        <div class="rank-badge">#${index + 1}</div>
        <div class="ranked-details">
          <div class="score-breakdown">
            <span class="score-chip final">Score: ${item.finalScore.toFixed(3)}</span>
            <span class="score-chip semantic">Sim(&alpha;=${alpha.toFixed(2)}): ${item.sim.toFixed(2)}</span>
            <span class="score-chip decay">Decay(&lambda;=${lambda.toFixed(1)}): ${item.timeDecay.toFixed(2)} (${item.chunk.daysAgo}d ago)</span>
            <span class="badge badge-violet" style="margin-left: auto;">${item.chunk.topic}</span>
          </div>
          <div style="font-size: 0.88rem; color: #e2e8f0; font-family: var(--font-mono); background: rgba(0,0,0,0.25); padding: 0.75rem; border-radius: 6px; white-space: pre-wrap;">${item.chunk.attributedText}</div>
        </div>
      `;
      container.appendChild(card);
    });
  }

  function updateGroundedAnswer(queryKey, topCandidate) {
    const answerElem = document.getElementById('grounded-answer-content');
    const citationContainer = document.getElementById('citation-cards-container');
    if (!answerElem || !citationContainer) return;

    citationContainer.innerHTML = '';

    let answerText = "";
    if (queryKey === "frontend") {
      if (topCandidate.chunk.id === "CTX-CHUNK-06") {
        answerText = `Based on the latest update on <strong>August 15, 2026</strong>, <strong>Rahul</strong> announced that following suggestions from project guide <strong>Ms. Bindhya Bhadran</strong>, the team changed their plans and will use <strong>Flutter</strong> instead of React to support both Android and iOS mobile platforms.`;
      } else {
        answerText = `On <strong>January 10, 2026</strong>, <strong>Rahul</strong> and <strong>Sai</strong> agreed to finalize <strong>React</strong> for the web dashboard (<em>Warning: This is an older decision superseded by August discussions!</em>).`;
      }
    } else if (queryKey === "documents") {
      answerText = `In the guide review discussion on <strong>August 12, 2026</strong>, <strong>Rahul</strong> explicitly committed: <em>"I will bring the documents tomorrow"</em>, with meeting time confirmed around 4 PM by Sai.`;
    } else if (queryKey === "trip") {
      answerText = `The group agreed to go to <strong>Goa in December</strong> for their semester break trip, with tickets to be booked next month, as proposed by Dheeraj and finalized by Rahul.`;
    } else if (queryKey === "assignment") {
      answerText = `Neither Sai nor Dheeraj had finished the ML assignment due that night (Sai was currently working on question 3).`;
    }

    answerElem.innerHTML = answerText;

    // Add citation card
    const citCard = document.createElement('div');
    citCard.className = 'citation-card verified';
    citCard.innerHTML = `
      <div style="display: flex; justify-content: space-between; align-items: center;">
        <span class="badge badge-emerald">Verified Chat Evidence</span>
        <span style="font-size: 0.72rem; color: var(--text-muted);">${topCandidate.chunk.id}</span>
      </div>
      <div style="font-size: 0.85rem; color: #cbd5e1; font-family: var(--font-mono); margin-top: 0.4rem; white-space: pre-wrap;">${topCandidate.chunk.attributedText}</div>
      <div style="font-size: 0.72rem; color: var(--accent-cyan); margin-top: 0.3rem;">
        Participants: ${topCandidate.chunk.participants.join(', ')} • Timestamp: ${topCandidate.chunk.timeDisplay}
      </div>
    `;
    citationContainer.appendChild(citCard);
  }

  // Event Listeners for UI
  document.getElementById('btn-toggle-system')?.addEventListener('click', (e) => {
    filterSystemNotices = !filterSystemNotices;
    e.target.innerText = `Filter System Notices (${filterSystemNotices ? 'ON' : 'OFF'})`;
    e.target.className = filterSystemNotices ? 'badge badge-cyan' : 'badge badge-amber';
    renderChatStream();
  });

  document.getElementById('btn-reload-stream')?.addEventListener('click', () => {
    renderChatStream();
  });

  const sliderAlpha = document.getElementById('slider-alpha');
  sliderAlpha?.addEventListener('input', (e) => {
    alpha = parseFloat(e.target.value);
    document.getElementById('val-alpha').innerText = alpha.toFixed(2);
    calculateRankings();
  });

  const sliderLambda = document.getElementById('slider-lambda');
  sliderLambda?.addEventListener('input', (e) => {
    lambda = parseFloat(e.target.value);
    document.getElementById('val-lambda').innerText = lambda.toFixed(2);
    calculateRankings();
  });

  const sliderTopK = document.getElementById('slider-topk');
  sliderTopK?.addEventListener('input', (e) => {
    topK = parseInt(e.target.value);
    document.getElementById('val-topk').innerText = topK;
    calculateRankings();
  });

  document.getElementById('btn-run-query')?.addEventListener('click', () => {
    const input = document.getElementById('query-input');
    if (input && input.value.trim()) {
      currentQuery = input.value.trim();
      calculateRankings();
    }
  });

  document.getElementById('query-input')?.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
      currentQuery = e.target.value.trim();
      calculateRankings();
    }
  });

  document.querySelectorAll('.sample-query-pill').forEach(pill => {
    pill.addEventListener('click', () => {
      const q = pill.getAttribute('data-query');
      if (q) {
        currentQuery = q;
        const input = document.getElementById('query-input');
        if (input) input.value = q;
        calculateRankings();
      }
    });
  });

  // Initial Boot
  renderChatStream();
  renderChunkComparison();
  calculateRankings();
});
