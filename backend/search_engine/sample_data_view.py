from rest_framework.views import APIView
from rest_framework.response import Response
from .models import Document
import json
from datetime import datetime

class LoadSampleDataAPIView(APIView):
    def post(self, request):
        """Load sample publications for testing/demo purposes"""
        
        # Clear existing data
        Document.objects.all().delete()
        
        sample_data = [
            {
                "title": "Machine Learning Approaches for Mathematics and Data Analysis",
                "url": "https://pureportal.coventry.ac.uk/en/publications/machine-learning-mathematics-2023",
                "authors": json.dumps(["Dr. John Smith", "Dr. Jane Doe", "Prof. Michael Johnson"]),
                "year": "2023",
                "abstract": "This paper presents novel machine learning algorithms for mathematical data analysis and computational modeling.",
                "keywords": "machine learning, mathematics, data analysis",
                "profile_link": "https://pureportal.coventry.ac.uk/en/persons/john-smith"
            },
            {
                "title": "Advanced Neural Networks and Deep Learning in Applied Mathematics",
                "url": "https://pureportal.coventry.ac.uk/en/publications/neural-networks-mathematics-2023",
                "authors": json.dumps(["Prof. Michael Johnson", "Dr. Sarah Williams"]),
                "year": "2023",
                "abstract": "Deep learning methods for solving differential equations and complex mathematical problems in scientific computing.",
                "keywords": "neural networks, deep learning, mathematics",
                "profile_link": "https://pureportal.coventry.ac.uk/en/persons/michael-johnson"
            },
            {
                "title": "Computational Mathematics: Algorithms and Scientific Applications",
                "url": "https://pureportal.coventry.ac.uk/en/publications/computational-mathematics-2022",
                "authors": json.dumps(["Dr. Robert Brown", "Dr. Emily Davis"]),
                "year": "2022",
                "abstract": "Computational methods for solving complex mathematical and scientific problems using advanced algorithms.",
                "keywords": "computational mathematics, algorithms, scientific computing",
                "profile_link": "https://pureportal.coventry.ac.uk/en/persons/robert-brown"
            },
            {
                "title": "Mathematical Modelling in Scientific Research and Engineering",
                "url": "https://pureportal.coventry.ac.uk/en/publications/mathematical-modelling-2022",
                "authors": json.dumps(["Prof. Andrew Wilson", "Dr. Lisa Anderson"]),
                "year": "2022",
                "abstract": "Mathematical models and modelling techniques for understanding physical phenomena in engineering applications.",
                "keywords": "mathematical modelling, engineering, research",
                "profile_link": "https://pureportal.coventry.ac.uk/en/persons/andrew-wilson"
            },
            {
                "title": "Big Data Processing and Mathematical Analysis Techniques",
                "url": "https://pureportal.coventry.ac.uk/en/publications/big-data-mathematics-2023",
                "authors": json.dumps(["Dr. David Martinez", "Dr. Jennifer Lee"]),
                "year": "2023",
                "abstract": "Scalable techniques for processing massive datasets with mathematical analysis in distributed systems.",
                "keywords": "big data, data processing, mathematical analysis",
                "profile_link": "https://pureportal.coventry.ac.uk/en/persons/david-martinez"
            },
            {
                "title": "Optimization Methods for Mathematical Models in Machine Learning",
                "url": "https://pureportal.coventry.ac.uk/en/publications/optimization-mathematics-2024",
                "authors": json.dumps(["Dr. Jane Doe", "Prof. Michael Johnson"]),
                "year": "2024",
                "abstract": "Advanced mathematical optimization techniques for training neural networks and ML models efficiently.",
                "keywords": "optimization, machine learning, mathematics",
                "profile_link": "https://pureportal.coventry.ac.uk/en/persons/jane-doe"
            },
            {
                "title": "Numerical Methods in Mathematics for Scientific Computing",
                "url": "https://pureportal.coventry.ac.uk/en/publications/numerical-methods-2023",
                "authors": json.dumps(["Dr. Peter Chen", "Prof. Richard Thomas"]),
                "year": "2023",
                "abstract": "Numerical mathematical methods for scientific computing and complex problem solving in research.",
                "keywords": "numerical methods, scientific computing, mathematics",
                "profile_link": "https://pureportal.coventry.ac.uk/en/persons/peter-chen"
            },
            {
                "title": "Advanced Statistics and Mathematical Analysis in Data Science",
                "url": "https://pureportal.coventry.ac.uk/en/publications/statistics-mathematics-2024",
                "authors": json.dumps(["Dr. Sophie Martin", "Dr. Kevin White"]),
                "year": "2024",
                "abstract": "Mathematical and statistical approaches for advanced data science and predictive analysis.",
                "keywords": "statistics, data science, mathematical analysis",
                "profile_link": "https://pureportal.coventry.ac.uk/en/persons/sophie-martin"
            },
            {
                "title": "Artificial Intelligence in Healthcare: A Comprehensive Review",
                "url": "https://pureportal.coventry.ac.uk/en/publications/ai-healthcare-review-2025",
                "authors": json.dumps(["Dr. Emma Thompson", "Prof. James Wilson"]),
                "year": "2025",
                "abstract": "A comprehensive review of artificial intelligence applications in modern healthcare systems.",
                "keywords": "artificial intelligence, healthcare, medical AI",
                "profile_link": "https://pureportal.coventry.ac.uk/en/persons/emma-thompson"
            },
            {
                "title": "Deep Learning Approaches for Natural Language Processing",
                "url": "https://pureportal.coventry.ac.uk/en/publications/deep-learning-nlp-2024",
                "authors": json.dumps(["Dr. Alex Kumar", "Dr. Maria Garcia"]),
                "year": "2024",
                "abstract": "State-of-the-art deep learning techniques for natural language understanding and generation.",
                "keywords": "deep learning, NLP, natural language processing",
                "profile_link": "https://pureportal.coventry.ac.uk/en/persons/alex-kumar"
            }
        ]
        
        # Create documents
        for item in sample_data:
            Document.objects.create(**item)
        
        # Rebuild index
        from .services import SearchEngine
        SearchEngine.get_instance().build_index()
        
        return Response({
            "status": "success",
            "message": f"Loaded {len(sample_data)} sample publications",
            "count": len(sample_data)
        })
