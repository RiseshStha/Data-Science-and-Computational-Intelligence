import { useState } from 'react';
import { Link } from 'react-router-dom';
import axios from 'axios';

const ClusteringPage = () => {
    const [inputText, setInputText] = useState('');
    const [prediction, setPrediction] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    const API_BASE_URL = 'http://localhost:8000/api/clustering';

    // 1. Cluster-based styling (No Categories)
    const clusterColors = {
        0: 'bg-blue-100 text-blue-800 border-blue-300',
        1: 'bg-purple-100 text-purple-800 border-purple-300',
        2: 'bg-green-100 text-green-800 border-green-300',
    };

    const handlePredict = async () => {
        if (!inputText.trim()) {
            setError('Please enter some text');
            return;
        }

        setLoading(true);
        setError(null);
        setPrediction(null);

        try {
            // Updated endpoint to 'assign' (academic terminology)
            const response = await axios.post(`${API_BASE_URL}/assign/`, {
                text: inputText,
            });

            setPrediction(response.data);
        } catch (err) {
            setError(err.response?.data?.error || 'Failed to assign cluster');
            console.error('Prediction error:', err);
        } finally {
            setLoading(false);
        }
    };

    // 6. Label-Agnostic Test Examples (No color hints)
    const testExamples = [
        {
            text: 'Inflation rates hit a new high as corporate profits soar in the global market.'
        },
        {
            text: 'The new blockbuster film starring the famous actor won an Oscar for best director.'
        },
        {
            text: 'NHS hospitals are facing severe delays as doctors and nurses treat more patients.'
        },
    ];

    const handleExampleClick = (example) => {
        setInputText(example.text);
        setPrediction(null);
        setError(null);
    };

    return (
        <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100">
            {/* Navigation */}
            <nav className="bg-white shadow-md border-b border-slate-200">
                <div className="max-w-7xl mx-auto px-6 py-4">
                    <div className="flex items-center justify-between">
                        <div className="flex items-center space-x-8">
                            <h1 className="text-2xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                                Research Publication
                            </h1>
                            <div className="flex space-x-4">
                                <Link
                                    to="/"
                                    className="px-4 py-2 text-slate-600 hover:text-slate-900 hover:bg-slate-100 rounded-lg transition-all"
                                >
                                    🔍 Search Engine
                                </Link>
                                <Link
                                    to="/clustering"
                                    className="px-4 py-2 bg-gradient-to-r from-blue-500 to-purple-500 text-white rounded-lg font-medium shadow-md"
                                >
                                    📊 Document Clustering
                                </Link>
                                <Link
                                    to="/crawl"
                                    className="px-4 py-2 text-slate-600 hover:text-slate-900 hover:bg-slate-100 rounded-lg transition-all"
                                >
                                    🕷️ Crawler
                                </Link>
                                <Link
                                    to="/stats"
                                    className="px-4 py-2 text-slate-600 hover:text-slate-900 hover:bg-slate-100 rounded-lg transition-all"
                                >
                                    📈 Statistics
                                </Link>
                            </div>
                        </div>
                    </div>
                </div>
            </nav>

            {/* Main Content */}
            <div className="max-w-5xl mx-auto px-6 py-12">
                {/* Header */}
                <div className="text-center mb-12">
                    <h2 className="text-4xl font-bold text-slate-900 mb-4">
                        Document Clustering
                    </h2>
                    <p className="text-lg text-slate-600 max-w-2xl mx-auto">
                        Enter any text and it will <span className="font-semibold text-blue-600">assign the document to one of the discovered clusters</span> based on its content.
                    </p>
                </div>

                {/* Input Section */}
                <div className="bg-white rounded-2xl shadow-xl p-8 mb-8 border border-slate-200">
                    <label className="block text-sm font-semibold text-slate-700 mb-3">
                        Enter Document Text
                    </label>
                    <textarea
                        value={inputText}
                        onChange={(e) => setInputText(e.target.value)}
                        placeholder="Type or paste your text here... (e.g., 'The stock market rallied today as investors welcomed strong corporate earnings')"
                        className="w-full h-40 px-4 py-3 border-2 border-slate-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none text-slate-900 placeholder-slate-400"
                    />

                    <div className="flex items-center justify-between mt-6">
                        <div className="text-sm text-slate-500">
                            {inputText.length} characters
                        </div>
                        <div className="flex space-x-3">
                            <button
                                onClick={() => {
                                    setInputText('');
                                    setPrediction(null);
                                    setError(null);
                                }}
                                className="px-6 py-2.5 bg-slate-100 text-slate-700 rounded-lg hover:bg-slate-200 transition-all font-medium"
                            >
                                Clear
                            </button>
                            <button
                                onClick={handlePredict}
                                disabled={loading || !inputText.trim()}
                                className="px-8 py-2.5 bg-gradient-to-r from-blue-500 to-purple-500 text-white rounded-lg hover:from-blue-600 hover:to-purple-600 disabled:opacity-50 disabled:cursor-not-allowed transition-all font-medium shadow-md hover:shadow-lg"
                            >
                                {loading ? (
                                    <span className="flex items-center">
                                        <svg className="animate-spin -ml-1 mr-2 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                                            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                                            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                                        </svg>
                                        Analyzing...
                                    </span>
                                ) : (
                                    '🔗 Assign to Cluster'
                                )}
                            </button>
                        </div>
                    </div>
                </div>

                {/* Error Message */}
                {error && (
                    <div className="bg-red-50 border-l-4 border-red-500 p-4 mb-8 rounded-lg">
                        <div className="flex items-center">
                            <span className="text-red-600 font-medium">❌ {error}</span>
                        </div>
                    </div>
                )}

                {/* Prediction Results */}
                {prediction && !prediction.error && (
                    <div className="bg-white rounded-2xl shadow-xl p-8 mb-8 border border-slate-200 animate-fadeIn">
                        <h3 className="text-2xl font-bold text-slate-900 mb-6">
                            Clustering Results
                        </h3>

                        {/* Main Result Display */}
                        <div className={`p-6 rounded-xl border-2 mb-6 ${clusterColors[prediction.cluster] || 'bg-slate-100 border-slate-300'}`}>
                            <div className="flex items-center justify-between">
                                <div>
                                    <div className="text-sm font-medium opacity-75 uppercase tracking-wide">Assigned Cluster</div>
                                    <div className="text-4xl font-bold mt-1">
                                        Cluster {prediction.cluster}
                                    </div>
                                </div>
                                <div className="text-right">
                                    <div className="text-sm font-medium opacity-75">Similarity Score</div>
                                    <div className="text-3xl font-bold">{(prediction.similarity_score).toFixed(4)}</div>
                                </div>
                            </div>
                        </div>

                        {/* Cluster Interpretation / Keywords */}
                        {prediction.top_terms && (
                            <div className="mt-8">
                                <h4 className="text-lg font-semibold text-slate-700 mb-3">
                                    Cluster Interpretation (Dominant Terms)
                                </h4>
                                <div className="flex flex-wrap gap-2">
                                    {prediction.top_terms.map((term, index) => (
                                        <span
                                            key={index}
                                            className="px-4 py-2 bg-slate-100 text-slate-700 rounded-lg text-sm font-medium border border-slate-200"
                                        >
                                            {term}
                                        </span>
                                    ))}
                                </div>
                                <p className="text-sm text-slate-500 mt-3 italic">
                                    These are the most significant terms defining this cluster, automatically extracted from the dataset.
                                </p>
                            </div>
                        )}
                    </div>
                )}

                {/* Example Tests */}
                <div className="bg-white rounded-2xl shadow-xl p-8 border border-slate-200">
                    <h3 className="text-xl font-bold text-slate-900 mb-4">
                        Try These Examples
                    </h3>
                    <p className="text-slate-600 mb-6">
                        Click on any example to test the clustering model
                    </p>
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                        {testExamples.map((example, index) => (
                            <button
                                key={index}
                                onClick={() => handleExampleClick(example)}
                                className={`p-4 rounded-xl border-2 border-slate-200 bg-slate-50 text-left hover:shadow-lg transition-all hover:scale-105 hover:bg-white hover:border-slate-300`}
                            >
                                <div className="flex items-center mb-2">
                                    <span className="font-semibold text-lg text-slate-800">Example {index + 1}</span>
                                </div>
                                <p className="text-sm opacity-90 text-slate-600">{example.text}</p>
                            </button>
                        ))}
                    </div>
                </div>
            </div>
        </div>
    );
};

export default ClusteringPage;
