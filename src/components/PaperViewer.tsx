'use client';

import React, { useState } from 'react';
import { Paper } from '@/types';
import { paperData, blogs } from '@/data/papers';

interface PaperViewerProps {
  paper?: Paper;
}

export function PaperViewer({ paper = paperData }: PaperViewerProps) {
  const [expandedSection, setExpandedSection] = useState<number | null>(0);
  const [showBibtexModal, setShowBibtexModal] = useState<boolean>(false);
  const [copiedBibtex, setCopiedBibtex] = useState<boolean>(false);

  const handleCopyBibtex = () => {
    navigator.clipboard.writeText(paper.bibtex);
    setCopiedBibtex(true);
    setTimeout(() => setCopiedBibtex(false), 2000);
  };

  return (
    <div className="paper-viewer-container">
      {/* Paper Header Box */}
      <div className="p-6 bg-[#13120f] rounded-2xl border border-[#2a2825] mb-6">
        <div className="flex flex-wrap items-center justify-between gap-3 mb-3">
          <span className="text-xs font-mono px-3 py-1 bg-[#E53935]/15 text-[#E53935] border border-[#E53935]/30 rounded-full font-semibold">
            {paper.doi}
          </span>
          <button
            onClick={() => setShowBibtexModal(true)}
            className="text-xs font-mono px-3 py-1.5 bg-[#1a1916] hover:bg-[#2a2825] text-white border border-[#2a2825] rounded-lg transition-colors flex items-center gap-1.5"
          >
            <span>Cite Paper (BibTeX)</span>
          </button>
        </div>

        <h1 className="text-2xl font-mono font-bold text-white mb-2 leading-tight">
          {paper.title}
        </h1>

        <div className="flex flex-wrap items-center gap-4 text-xs font-mono text-[#A09D96] mb-4">
          <span className="text-white font-medium">{paper.authors}</span>
          <span>&bull;</span>
          <span>{paper.date}</span>
        </div>

        {/* Abstract */}
        <div className="p-4 bg-[#1a1916] rounded-xl border border-[#2a2825] mb-4">
          <span className="text-xs font-mono text-[#78909C] uppercase tracking-wider block mb-2 font-bold">
            Abstract
          </span>
          <p className="text-xs text-[#F5F5F0] leading-relaxed">
            {paper.abstract}
          </p>
        </div>

        {/* Tags */}
        <div className="flex flex-wrap gap-2">
          {paper.tags.map((tag) => (
            <span
              key={tag}
              className="text-[11px] font-mono px-2.5 py-0.5 bg-[#100f0c] text-[#A09D96] border border-[#2a2825] rounded-md"
            >
              #{tag}
            </span>
          ))}
        </div>
      </div>

      {/* Expandable Paper Sections */}
      <div className="p-6 bg-[#13120f] rounded-2xl border border-[#2a2825] mb-8">
        <h2 className="text-base font-mono font-bold text-white mb-4 uppercase tracking-wider">
          Complete Research Manuscript Sections
        </h2>
        <div className="flex flex-col gap-3">
          {paper.sections.map((section, idx) => {
            const isExpanded = expandedSection === idx;
            return (
              <div
                key={idx}
                className="rounded-xl border border-[#2a2825] overflow-hidden bg-[#1a1916]"
              >
                <button
                  onClick={() => setExpandedSection(isExpanded ? null : idx)}
                  className="w-full px-4 py-3 text-left flex items-center justify-between text-xs font-mono font-semibold text-white hover:bg-[#22211d] transition-colors"
                >
                  <span>{section.title}</span>
                  <span className="text-[#E53935] text-sm">{isExpanded ? '−' : '+'}</span>
                </button>
                {isExpanded && (
                  <div className="px-4 py-3.5 text-xs text-[#A09D96] leading-relaxed border-t border-[#2a2825] bg-[#13120f] whitespace-pre-line font-sans">
                    {section.content}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>

      {/* Engineering Deep Dives & Articles */}
      <div className="mb-6">
        <h3 className="text-lg font-mono font-bold text-white mb-4 uppercase tracking-wider">
          Systems Engineering Deep Dives
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {blogs.map((b) => (
            <div key={b.id} className="p-4 bg-[#13120f] rounded-xl border border-[#2a2825] flex flex-col justify-between">
              <div>
                <div className="flex items-center justify-between text-[11px] font-mono text-[#78909C] mb-2">
                  <span>{b.date}</span>
                  <span>{b.readTime}</span>
                </div>
                <h4 className="text-sm font-mono font-bold text-white mb-2 leading-snug">
                  {b.title}
                </h4>
                <p className="text-xs text-[#A09D96] line-clamp-3 mb-3">
                  {b.excerpt}
                </p>
              </div>
              <div className="flex flex-wrap gap-1.5 pt-2 border-t border-[#2a2825]">
                {b.tags.map((t) => (
                  <span key={t} className="text-[10px] font-mono px-2 py-0.5 bg-[#1a1916] text-[#A09D96] rounded">
                    {t}
                  </span>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* BibTeX Citation Modal */}
      {showBibtexModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm animate-in fade-in">
          <div className="w-full max-w-xl bg-[#13120f] border border-[#2a2825] rounded-2xl p-6 shadow-2xl relative">
            <div className="flex items-center justify-between pb-3 mb-4 border-b border-[#2a2825]">
              <div className="flex items-center gap-2">
                <span className="w-3 h-3 rounded-full bg-[#E53935]"></span>
                <h3 className="font-mono text-sm font-bold text-white">BibTeX Citation Entry</h3>
              </div>
              <button
                onClick={() => setShowBibtexModal(false)}
                className="text-[#A09D96] hover:text-white font-mono text-sm px-2 py-1 rounded"
              >
                ✕
              </button>
            </div>

            <pre className="p-4 bg-[#0a0a08] border border-[#2a2825] rounded-xl text-xs font-mono text-[#4CAF50] overflow-x-auto mb-4 leading-relaxed">
              <code>{paper.bibtex}</code>
            </pre>

            <div className="flex items-center justify-end gap-3">
              <button
                onClick={() => setShowBibtexModal(false)}
                className="px-4 py-2 text-xs font-mono text-[#A09D96] hover:text-white rounded-lg border border-[#2a2825]"
              >
                Close
              </button>
              <button
                onClick={handleCopyBibtex}
                className="px-4 py-2 text-xs font-mono bg-[#E53935] hover:bg-[#B71C1C] text-white font-semibold rounded-lg transition-colors flex items-center gap-2"
              >
                {copiedBibtex ? (
                  <>
                    <span className="text-white">✓</span> Copied to Clipboard
                  </>
                ) : (
                  <>Copy BibTeX Citation</>
                )}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default PaperViewer;
