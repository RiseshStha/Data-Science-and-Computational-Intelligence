import React, { useState, useRef } from 'react';
import { Play, Terminal, ArrowLeft, Loader2, Link as LinkIcon, Database } from 'lucide-react';
import { Link } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';

const CrawlPage = () => {
    const [url, setUrl] = useState('');
    const [logs, setLogs] = useState([]);
    const [isCrawling, setIsCrawling] = useState(false);

    // Auto-scroll ref
    const logContainerRef = useRef(null);

    const appendLog = (message) => {
        setLogs(prev => [...prev, message]);
        if (logContainerRef.current) {
            logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight;
        }
    };

    const [discoveredUrls, setDiscoveredUrls] = useState([]);

    const handleCrawl = async (e) => {
        e.preventDefault();
        if (!url) return;

        setIsCrawling(true);
        setLogs([]);
        setDiscoveredUrls([]); // Reset
        appendLog(`Initializing crawl for: ${url}...`);

        try {
            const response = await fetch('http://127.0.0.1:8000/api/crawl/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ url: url })
            });

            const reader = response.body.getReader();
            const decoder = new TextDecoder();

            while (true) {
                const { value, done } = await reader.read();
                if (done) break;

                const text = decoder.decode(value);
                const lines = text.split('\n');

                for (const line of lines) {
                    if (line.trim()) {
                        appendLog(line);

                        // Extract URL if present
                        const urlMatch = line.match(/(https?:\/\/[^\s]+)/g);
                        if (urlMatch) {
                            urlMatch.forEach(u => {
                                // Add only unique URLs that look like content pages
                                setDiscoveredUrls(prev => {
                                    if (prev.includes(u)) return prev;
                                    return [u, ...prev].slice(0, 10); // Keep last 10
                                });
                            });
                        }
                    }
                }
            }
        } catch (error) {
            appendLog(`Error: ${error.message}`);
        } finally {
            setIsCrawling(false);
            appendLog('Process terminated.');
        }
    };

    const loadSampleData = () => {
        const SAMPLE_URL = "https://pureportal.coventry.ac.uk/en/organisations/ics-research-centre-for-computational-science-and-mathematical-mo";
        setUrl(SAMPLE_URL);
    };

    const handleLoadSampleData = async () => {
        setIsCrawling(true);
        setLogs([]);
        appendLog('Loading sample publications...');

        try {
            const response = await fetch('http://127.0.0.1:8000/api/load-sample-data/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                }
            });

            const data = await response.json();

            if (data.status === 'success') {
                appendLog(`✓ Successfully loaded ${data.count} sample publications`);
                appendLog('✓ Index rebuilt automatically');
                appendLog('✓ Ready to search!');
                appendLog('');
                appendLog('Sample data includes publications on:');
                appendLog('  - Machine Learning & AI');
                appendLog('  - Mathematical Modeling');
                appendLog('  - Data Science & Analytics');
                appendLog('  - Neural Networks & Deep Learning');
                appendLog('');
                appendLog('Try searching for: "machine learning", "mathematics", or "data"');
            } else {
                appendLog(`Error: ${data.message || 'Failed to load sample data'}`);
            }
        } catch (error) {
            appendLog(`Error: ${error.message}`);
        } finally {
            setIsCrawling(false);
        }
    };

    return (
        <div className="min-h-screen flex flex-col items-center py-12 px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto transition-all duration-500">

            {/* Navigation to Search */}
            <div className="absolute top-4 left-4 z-20">
                <Link
                    to="/"
                    className="flex items-center space-x-2 text-sm text-slate-500 hover:text-primary transition-colors bg-white/50 backdrop-blur-sm px-3 py-1.5 rounded-full border border-gray-200 hover:border-primary/30 hover:shadow-sm"
                >
                    <ArrowLeft size={16} />
                    <span className="font-medium">Back to Search</span>
                </Link>
            </div>

            {/* Navigation to Stats */}
            <div className="absolute top-4 right-4 z-20">
                <Link
                    to="/stats"
                    className="flex items-center space-x-2 text-sm text-slate-500 hover:text-primary transition-colors bg-white/50 backdrop-blur-sm px-3 py-1.5 rounded-full border border-gray-200 hover:border-primary/30 hover:shadow-sm"
                >
                    <span className="font-medium">Statistics</span>
                </Link>
            </div>

            {/* Header / Logo Area */}
            <motion.div
                layout
                className={`flex flex-col items-center mb-8 mt-16`}
            >
                <div className="flex items-center space-x-3 mb-4">
                    <Database size={48} className="text-secondary" />
                    <h1 className="text-4xl sm:text-5xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-secondary to-red-600">
                        Crawler Console
                    </h1>
                </div>
            </motion.div>

            {/* Search/Crawl Bar */}
            <motion.div layout className="w-full max-w-2xl relative z-10">
                <form onSubmit={handleCrawl} className="relative group">
                    <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
                        <LinkIcon className="h-5 w-5 text-gray-400 group-focus-within:text-secondary transition-colors" />
                    </div>
                    <input
                        type="text"
                        className="block w-full pl-12 pr-4 py-4 border-2 border-gray-200 rounded-full leading-5 bg-white placeholder-gray-400 focus:outline-none focus:border-secondary focus:ring-4 focus:ring-secondary/10 transition-all duration-300 shadow-sm hover:shadow-md"
                        placeholder="Enter URL to crawl (e.g., https://example.com/research)..."
                        value={url}
                        onChange={(e) => setUrl(e.target.value)}
                    />
                    <div className="absolute inset-y-0 right-0 pr-3 flex items-center">
                        <button
                            type="submit"
                            className="p-2 bg-secondary text-white rounded-full hover:bg-red-600 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-secondary transition-colors disabled:opacity-50"
                            disabled={isCrawling || !url}
                        >
                            {isCrawling ? <Loader2 className="h-5 w-5 animate-spin" /> : <Play className="h-5 w-5 ml-0.5" />}
                        </button>
                    </div>
                </form>

                <div className="flex justify-center mt-3 space-x-4">
                    <button
                        onClick={loadSampleData}
                        className="text-xs text-slate-400 hover:text-secondary transition-colors"
                    >
                        Load Sample URL (Coventry Portal)
                    </button>
                    <span className="text-slate-300">|</span>
                    <button
                        onClick={handleLoadSampleData}
                        disabled={isCrawling}
                        className="text-xs text-slate-400 hover:text-green-600 transition-colors disabled:opacity-50 font-medium"
                    >
                        ⚡ Load Sample Publications 
                    </button>
                </div>
            </motion.div>

            {/* Live Discovery Feed */}
            <AnimatePresence>
                {discoveredUrls.length > 0 && (
                    <motion.div
                        initial={{ opacity: 0, height: 0 }}
                        animate={{ opacity: 1, height: 'auto' }}
                        exit={{ opacity: 0, height: 0 }}
                        className="w-full max-w-4xl mb-4"
                    >
                        <div className="flex gap-2 overflow-x-auto pb-4 pt-2 px-2 scrollbar-hide">
                            {discoveredUrls.map((u, i) => (
                                <motion.a
                                    key={i}
                                    initial={{ scale: 0.8, opacity: 0 }}
                                    animate={{ scale: 1, opacity: 1 }}
                                    href={u}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    className="flex-shrink-0 bg-white shadow-sm border border-slate-200 rounded-lg p-3 w-64 text-xs group hover:border-secondary hover:shadow-md transition-all text-decoration-none"
                                >
                                    <div className="flex items-center gap-2 mb-1 text-slate-400 group-hover:text-secondary">
                                        <div className="w-1.5 h-1.5 rounded-full bg-green-500 animate-pulse"></div>
                                        <span className="font-bold uppercase tracking-wider text-[10px]">Discovered</span>
                                    </div>
                                    <div className="truncate font-medium text-slate-700">{u.split('/').pop() || 'Page'}</div>
                                    <div className="truncate text-[10px] text-slate-400 mt-0.5">{u}</div>
                                </motion.a>
                            ))}
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>

            {/* Logs Window */}
            <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="w-full max-w-4xl mt-8"
            >
                <div className="bg-slate-900 rounded-xl shadow-lg border border-slate-800 overflow-hidden flex flex-col h-[500px]">
                    <div className="bg-slate-800 px-4 py-2 border-b border-slate-700 flex items-center justify-between">
                        <div className="flex items-center space-x-2 text-slate-400 text-xs uppercase tracking-wider font-semibold">
                            <Terminal size={14} />
                            <span>System Output</span>
                        </div>
                        {isCrawling && (
                            <div className="flex items-center gap-2 text-xs text-green-400">
                                <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse"></div>
                                Running
                            </div>
                        )}
                    </div>

                    <div
                        ref={logContainerRef}
                        className="flex-1 p-4 font-mono text-sm overflow-y-auto space-y-1"
                    >
                        {logs.length === 0 ? (
                            <div className="h-full flex flex-col items-center justify-center text-slate-700 space-y-2 opacity-50">
                                <Terminal size={32} />
                                <p>Ready to initialize crawler...</p>
                            </div>
                        ) : (
                            logs.map((log, i) => (
                                <div key={i} className="break-all">
                                    <span className="text-slate-600 mr-2 select-none">[{new Date().toLocaleTimeString()}]</span>
                                    <span className={log.includes('Error') ? 'text-red-400' : 'text-slate-300'}>
                                        {log}
                                    </span>
                                </div>
                            ))
                        )}
                        {isCrawling && (
                            <div className="animate-pulse text-secondary text-xs mt-2">_</div>
                        )}
                    </div>
                </div>
            </motion.div>

            {/* Footer */}
            <div className="fixed bottom-0 w-full py-4 text-center text-xs text-gray-400 bg-slate-50/80 backdrop-blur-sm">
                <p>Research Publication Search • Powered by Django & React</p>
            </div>
        </div>
    );
};

export default CrawlPage;
