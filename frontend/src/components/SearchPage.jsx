import React, { useState } from 'react';
import axios from 'axios';
import { Search, BookOpen, Loader2, ExternalLink, GraduationCap } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { Link } from 'react-router-dom';

const SearchPage = () => {
    const [query, setQuery] = useState('');
    const [results, setResults] = useState([]);
    const [loading, setLoading] = useState(false);
    const [hasSearched, setHasSearched] = useState(false);

    const handleSearch = async (e) => {
        e.preventDefault();
        if (!query.trim()) return;

        setLoading(true);
        setHasSearched(true);
        try {
            const response = await axios.get(`http://127.0.0.1:8000/api/search/?q=${query}`);
            setResults(response.data.results);
        } catch (error) {
            console.error("Error fetching results:", error);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="min-h-screen flex flex-col items-center py-12 px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto transition-all duration-500">

            {/* Header / Logo Area */}
            <div className="absolute top-4 right-4 z-20 flex space-x-2">
                <Link
                    to="/clustering"
                    className="flex items-center space-x-2 text-sm text-slate-500 hover:text-primary transition-colors bg-white/50 backdrop-blur-sm px-3 py-1.5 rounded-full border border-gray-200 hover:border-primary/30 hover:shadow-sm"
                >
                    <span className="font-medium">📊 Clustering</span>
                </Link>
                <Link
                    to="/stats"
                    className="flex items-center space-x-2 text-sm text-slate-500 hover:text-primary transition-colors bg-white/50 backdrop-blur-sm px-3 py-1.5 rounded-full border border-gray-200 hover:border-primary/30 hover:shadow-sm"
                >
                    <span className="font-medium">Statistics</span>
                </Link>
                <Link
                    to="/crawl"
                    className="flex items-center space-x-2 text-sm text-slate-500 hover:text-primary transition-colors bg-white/50 backdrop-blur-sm px-3 py-1.5 rounded-full border border-gray-200 hover:border-primary/30 hover:shadow-sm"
                >
                    <span className="font-medium">Crawl Website</span>
                </Link>
            </div>

            <motion.div
                layout
                className={`flex flex-col items-center mb-8 ${hasSearched ? 'mt-0' : 'mt-32'}`}
            >
                <div className="flex items-center space-x-3 mb-4">
                    <GraduationCap size={48} className="text-primary" />
                    <h1 className="text-4xl sm:text-5xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-primary to-blue-600">
                        Research Publication Search
                    </h1>
                </div>
            </motion.div>

            {/* Search Bar */}
            <motion.div layout className="w-full max-w-2xl relative z-10">
                <form onSubmit={handleSearch} className="relative group">
                    <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
                        <Search className="h-5 w-5 text-gray-400 group-focus-within:text-primary transition-colors" />
                    </div>
                    <input
                        type="text"
                        className="block w-full pl-12 pr-4 py-4 border-2 border-gray-200 rounded-full leading-5 bg-white placeholder-gray-400 focus:outline-none focus:border-primary focus:ring-4 focus:ring-primary/10 transition-all duration-300 shadow-sm hover:shadow-md"
                        placeholder="Search for articles, theses, books, abstracts..."
                        value={query}
                        onChange={(e) => setQuery(e.target.value)}
                    />
                    <div className="absolute inset-y-0 right-0 pr-3 flex items-center">
                        <button
                            type="submit"
                            className="p-2 bg-primary text-white rounded-full hover:bg-blue-600 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary transition-colors disabled:opacity-50"
                            disabled={loading}
                        >
                            {loading ? <Loader2 className="h-5 w-5 animate-spin" /> : <Search className="h-5 w-5" />}
                        </button>
                    </div>
                </form>
            </motion.div>

            {/* Results Area */}
            <div className="w-full max-w-4xl mt-12">
                <AnimatePresence>
                    {hasSearched && (
                        <motion.div
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0 }}
                            className="space-y-6"
                        >
                            <div className="text-sm text-slate-500 mb-4 px-2">
                                Found {results.length} results for <span className="font-semibold text-slate-700">"{query}"</span>
                            </div>

                            {results.length === 0 && !loading ? (
                                <div className="text-center py-12 bg-white rounded-xl shadow-sm border border-gray-100">
                                    <BookOpen className="mx-auto h-12 w-12 text-gray-300" />
                                    <h3 className="mt-2 text-sm font-medium text-gray-900">No results found</h3>
                                    <p className="mt-1 text-sm text-gray-500">Try adjusting your search terms.</p>
                                </div>
                            ) : (
                                results.map((result, index) => (
                                    <motion.div
                                        key={index}
                                        initial={{ opacity: 0, y: 20 }}
                                        animate={{ opacity: 1, y: 0 }}
                                        transition={{ delay: index * 0.05 }}
                                        className="bg-white p-6 rounded-xl shadow-sm hover:shadow-md border border-gray-100 transition-all duration-200 group"
                                    >
                                        <div className="flex justify-between items-start">
                                            <div className="flex-1">
                                                <a
                                                    href={result.url}
                                                    target="_blank"
                                                    rel="noopener noreferrer"
                                                    className="block"
                                                >
                                                    <h3 className="text-xl font-semibold text-blue-700 group-hover:underline decoration-2 underline-offset-2 mb-1">
                                                        {result.title}
                                                    </h3>
                                                </a>

                                                {/* Meta Info Row */}
                                                <div className="flex flex-wrap items-center gap-2 mb-2 text-sm">
                                                    <span className="text-green-700 truncate max-w-xs">{result.url}</span>
                                                    {result.year && (
                                                        <>
                                                            <span className="text-gray-300">•</span>
                                                            <span className="bg-slate-100 text-slate-600 px-2 py-0.5 rounded text-xs font-medium">
                                                                {result.year}
                                                            </span>
                                                        </>
                                                    )}
                                                    {result.score && (
                                                        <span className="text-xs text-slate-400 ml-auto">Score: {result.score}</span>
                                                    )}
                                                </div>

                                                {/* Authors */}
                                                {result.authors && (
                                                    <div className="text-sm text-slate-800 font-medium mb-2">
                                                        {result.authors}
                                                    </div>
                                                )}

                                                {/* Abstract Snippet */}
                                                {result.abstract && (
                                                    <p className="text-slate-600 text-sm leading-relaxed mb-3 line-clamp-3">
                                                        {result.abstract}
                                                    </p>
                                                )}

                                                {/* Profile Link Option */}
                                                {result.profile_link && (
                                                    <div className="mt-2 text-xs">
                                                        <a href={result.profile_link} target="_blank" className="text-primary hover:underline">
                                                            View Author Profile
                                                        </a>
                                                    </div>
                                                )}
                                            </div>
                                            <a
                                                href={result.url}
                                                target="_blank"
                                                rel="noopener noreferrer"
                                                className="ml-4 p-2 text-gray-400 hover:text-primary transition-colors"
                                            >
                                                <ExternalLink size={20} />
                                            </a>
                                        </div>
                                        <div className="mt-4 flex items-center space-x-4 text-xs text-gray-500">
                                            <button className="hover:text-primary flex items-center space-x-1">
                                                <span>Cite</span>
                                            </button>
                                            <button className="hover:text-primary">Related articles</button>
                                        </div>
                                    </motion.div>
                                ))
                            )}
                        </motion.div>
                    )}
                </AnimatePresence>
            </div>

            {/* Footer */}
            <div className="fixed bottom-0 w-full py-4 text-center text-xs text-gray-400 bg-slate-50/80 backdrop-blur-sm flex justify-center items-center space-x-4">
                <p> Research Publication Search • Powered by Django & React</p>
                <div className="h-3 w-px bg-gray-300"></div>
                <Link
                    to="/crawl"
                    className="hover:text-primary transition-colors cursor-pointer"
                >
                    Data Console
                </Link>
            </div>
        </div>
    );
};

export default SearchPage;
