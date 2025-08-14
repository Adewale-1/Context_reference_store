#!/usr/bin/env python3
"""
TUI Dashboard Demo

Demonstrates the beautiful Terminal User Interface (TUI) Analytics Dashboard
for real-time monitoring of Context Reference Store performance.

Features demonstrated:
- Real-time performance metrics visualization
- Interactive compression analytics
- Memory usage monitoring with charts
- Color-coded alerts and recommendations
- Keyboard navigation between different tabs
- Live updating statistics and charts

Usage:
    python examples/tui_dashboard_demo.py

Controls (once running):
    ← → : Switch between tabs
    ↑ ↓ : Scroll within tabs
    R   : Force refresh metrics
    C   : Clear alerts
    Q   : Quit dashboard
"""

import json
import time
import random
import threading
from typing import List, Dict

try:
    import sys
    import os

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from context_store import ContextReferenceStore, CacheEvictionPolicy
    from context_store.monitoring import create_dashboard

    CONTEXT_STORE_AVAILABLE = True
except ImportError as e:
    print(f"Context Reference Store or monitoring not available: {e}")
    print("Run from the library root directory or install with: pip install -e .")
    CONTEXT_STORE_AVAILABLE = False

try:
    import curses

    CURSES_AVAILABLE = True
except ImportError:
    CURSES_AVAILABLE = False


def generate_realistic_workload_data() -> List[Dict]:
    """Generate realistic AI workload data for demonstration."""

    workload_samples = [
        # Large AI model responses
        {
            "type": "llm_response",
            "content": f"""
            Based on your query about sustainable energy solutions, I'll provide a comprehensive analysis
            of the current renewable energy landscape and emerging technologies that show promise for
            addressing climate change challenges.

            Solar Energy Developments:
            The solar photovoltaic industry has experienced unprecedented growth over the past decade,
            with efficiency improvements and cost reductions making solar power increasingly competitive
            with fossil fuels. Recent breakthroughs in perovskite tandem cells have achieved laboratory
            efficiencies exceeding 31%, while commercial silicon panels now routinely achieve 22-24%
            efficiency. The integration of artificial intelligence in solar panel manufacturing has
            improved quality control and reduced production costs by approximately 15%.

            Wind Power Innovations:
            Offshore wind technology represents one of the most promising frontiers in renewable energy.
            Modern offshore turbines with capacities exceeding 15MW are now operational, with blade
            diameters approaching 220 meters. These massive installations can generate enough electricity
            to power approximately 16,000 homes each. Advanced materials science has enabled the
            development of lighter, stronger composite materials that extend turbine lifespan while
            reducing maintenance requirements.

            Energy Storage Breakthroughs:
            The challenge of intermittency in renewable energy sources is being addressed through
            revolutionary advances in energy storage technology. Lithium-ion battery costs have
            declined by over 85% since 2010, making grid-scale storage economically viable. Emerging
            technologies such as iron-air batteries, compressed air energy storage, and advanced
            pumped hydro systems offer the potential for long-duration storage at scale.

            Smart Grid Integration:
            The modernization of electrical grids through smart technology enables better integration
            of distributed renewable energy sources. Machine learning algorithms optimize energy
            distribution in real-time, predicting demand patterns and automatically balancing supply
            from multiple renewable sources. Blockchain technology is facilitating peer-to-peer energy
            trading, allowing homeowners with solar panels to sell excess energy directly to neighbors.

            Policy and Economic Considerations:
            Government policies play a crucial role in accelerating renewable energy adoption.
            Feed-in tariffs, renewable energy certificates, and carbon pricing mechanisms create
            economic incentives for clean energy investments. The global renewable energy sector
            now employs over 13 million people worldwide, demonstrating the economic benefits of
            the green energy transition.

            Future Outlook:
            Projections indicate that renewable energy could supply 80% of global electricity demand
            by 2050, provided that current growth trends continue and supportive policies remain in
            place. The integration of hydrogen production from renewable sources offers additional
            pathways for decarbonizing heavy industry and transportation sectors.

            Technological convergence between renewable energy, energy storage, electric vehicles,
            and smart cities creates synergistic opportunities for comprehensive sustainability
            solutions. The continued advancement of these technologies, combined with decreasing
            costs and improving performance, positions renewable energy as the dominant energy
            source for the future global economy.
            """
            * random.randint(1, 3),
        },
        # API response data
        {
            "type": "api_response",
            "content": json.dumps(
                {
                    "users": [
                        {
                            "id": i,
                            "name": f"User {i}",
                            "email": f"user{i}@example.com",
                            "profile": {
                                "bio": f"Experienced software engineer with {random.randint(2, 15)} years in the industry. "
                                * 3,
                                "skills": [
                                    "Python",
                                    "JavaScript",
                                    "Machine Learning",
                                    "Cloud Computing",
                                ],
                                "projects": [
                                    {
                                        "name": f"Project {j}",
                                        "description": f"Advanced AI system for {random.choice(['healthcare', 'finance', 'education', 'transportation'])} applications. "
                                        * 5,
                                        "technologies": [
                                            "TensorFlow",
                                            "PyTorch",
                                            "Docker",
                                            "Kubernetes",
                                        ],
                                        "metrics": {
                                            "lines_of_code": random.randint(
                                                10000, 100000
                                            ),
                                            "commits": random.randint(100, 1000),
                                            "contributors": random.randint(5, 20),
                                        },
                                    }
                                    for j in range(random.randint(2, 5))
                                ],
                            },
                        }
                        for i in range(random.randint(50, 200))
                    ]
                }
            ),
        },
        # Code analysis data
        {
            "type": "code_analysis",
            "content": """
            import numpy as np
            import pandas as pd
            import tensorflow as tf
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import StandardScaler
            from sklearn.metrics import accuracy_score, classification_report
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            class AdvancedNeuralNetworkClassifier:
                '''
                Advanced neural network classifier with automatic hyperparameter tuning,
                regularization techniques, and comprehensive performance analysis.
                
                This implementation includes state-of-the-art techniques for preventing
                overfitting, optimizing training performance, and providing detailed
                model interpretability features.
                '''
                
                def __init__(self, input_dim, hidden_layers=[128, 64, 32], dropout_rate=0.3,
                           learning_rate=0.001, regularization=0.01):
                    '''
                    Initialize the advanced neural network classifier.
                    
                    Parameters:
                    -----------
                    input_dim : int
                        Dimensionality of input features
                    hidden_layers : list
                        List of hidden layer sizes
                    dropout_rate : float
                        Dropout rate for regularization
                    learning_rate : float
                        Learning rate for optimizer
                    regularization : float
                        L2 regularization strength
                    '''
                    self.input_dim = input_dim
                    self.hidden_layers = hidden_layers
                    self.dropout_rate = dropout_rate
                    self.learning_rate = learning_rate
                    self.regularization = regularization
                    self.model = None
                    self.history = None
                    self.scaler = StandardScaler()
                
                def build_model(self, num_classes):
                    '''
                    Build the neural network architecture with advanced regularization.
                    '''
                    model = tf.keras.Sequential()
                    
                    # Input layer
                    model.add(tf.keras.layers.Dense(
                        self.hidden_layers[0], 
                        input_dim=self.input_dim,
                        activation='relu',
                        kernel_regularizer=tf.keras.regularizers.l2(self.regularization)
                    ))
                    model.add(tf.keras.layers.BatchNormalization())
                    model.add(tf.keras.layers.Dropout(self.dropout_rate))
                    
                    # Hidden layers with progressive size reduction
                    for hidden_size in self.hidden_layers[1:]:
                        model.add(tf.keras.layers.Dense(
                            hidden_size,
                            activation='relu',
                            kernel_regularizer=tf.keras.regularizers.l2(self.regularization)
                        ))
                        model.add(tf.keras.layers.BatchNormalization())
                        model.add(tf.keras.layers.Dropout(self.dropout_rate))
                    
                    # Output layer
                    if num_classes == 2:
                        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
                        loss = 'binary_crossentropy'
                        metrics = ['accuracy']
                    else:
                        model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
                        loss = 'sparse_categorical_crossentropy'
                        metrics = ['accuracy']
                    
                    # Compile model with advanced optimizer
                    optimizer = tf.keras.optimizers.Adam(
                        learning_rate=self.learning_rate,
                        beta_1=0.9,
                        beta_2=0.999,
                        epsilon=1e-07
                    )
                    
                    model.compile(
                        optimizer=optimizer,
                        loss=loss,
                        metrics=metrics
                    )
                    
                    self.model = model
                    return model
            """
            * random.randint(2, 4),
        },
        # Documentation content
        {
            "type": "documentation",
            "content": f"""
            # Context Reference Store Documentation
            
            ## Overview
            
            The Context Reference Store is a revolutionary approach to managing large context windows
            in AI applications. By implementing a reference-based architecture, we achieve unprecedented
            efficiency gains while maintaining perfect content fidelity.
            
            ## Key Performance Metrics
            
            - **625x faster serialization** compared to traditional approaches
            - **49x memory reduction** in multi-agent scenarios  
            - **99.55% storage reduction** for multimodal content
            - **Zero quality degradation** validated with ROUGE metrics
            
            ## Architecture Components
            
            ### Core Storage Engine
            The storage engine implements advanced caching strategies with multiple eviction policies:
            
            - **LRU (Least Recently Used)**: Optimal for temporal locality patterns
            - **LFU (Least Frequently Used)**: Best for frequency-based access patterns
            - **TTL (Time To Live)**: Automatic expiration for time-sensitive content
            - **Memory Pressure**: Dynamic eviction based on system memory usage
            
            ### Multimodal Support
            Our hybrid storage architecture automatically handles different content types:
            
            - Small binaries (< 1MB) stored in memory for fast access
            - Large binaries (≥ 1MB) stored on disk with intelligent caching
            - SHA256-based deduplication prevents duplicate storage
            - Reference counting manages shared data lifecycle
            
            ### Advanced Caching Features
            
            #### Cache Warming
            The system intelligently identifies frequently accessed contexts and preloads them:
            
            ```python
            # Automatic warming based on access patterns
            context_store.warm_contexts([context_id_1, context_id_2])
            
            # Manual warming for critical contexts
            context_store.set_context_priority(context_id, priority=10)
            ```
            
            #### Background Processing
            TTL cleanup and maintenance operations run in background threads:
            
            - Automatic cleanup of expired contexts
            - Memory pressure monitoring and response
            - Statistics collection and analysis
            - Performance optimization recommendations
            
            ## Integration Examples
            
            ### Basic Usage
            
            ```python
            from context_store import ContextReferenceStore, CacheEvictionPolicy
            
            # Create context store with advanced configuration
            store = ContextReferenceStore(
                cache_size=100,
                eviction_policy=CacheEvictionPolicy.LRU,
                memory_threshold=0.8,
                enable_cache_warming=True
            )
            
            # Store large context efficiently
            context_id = store.store(large_document_text)
            
            # Retrieve with automatic caching optimization
            retrieved_content = store.retrieve(context_id)
            ```
            
            ### Framework Integration
            
            The Context Reference Store provides specialized adapters for popular AI frameworks:
            
            #### LangChain Integration
            ```python
            from context_store.adapters import LangChainContextAdapter
            
            adapter = LangChainContextAdapter(store)
            memory = adapter.create_conversation_memory()
            ```
            
            #### LangGraph Integration  
            ```python
            from context_store.adapters import LangGraphContextAdapter
            
            adapter = LangGraphContextAdapter(store)
            state_manager = adapter.create_state_manager()
            ```
            
            ## Performance Optimization
            
            ### Memory Management
            Monitor and optimize memory usage with built-in analytics:
            
            ```python
            # Get comprehensive statistics
            stats = store.get_cache_stats()
            print(f"Hit rate: {{stats['hit_rate']:.2%}}")
            print(f"Memory usage: {{stats['memory_usage_percent']:.1f}}%")
            
            # Get optimization recommendations
            recommendations = store.get_optimization_recommendations()
            for rec in recommendations:
                print(f"{{rec['priority']}}: {{rec['recommendation']}}")
            ```
            
            ### Compression Integration
            Enable smart compression for additional storage savings:
            
            ```python
            store = ContextReferenceStore(
                enable_compression=True,
                compression_min_size=1024  # Compress content > 1KB
            )
            
            # Get compression analytics
            compression_stats = store.get_compression_analytics()
            print(f"Space savings: {{compression_stats['space_savings_percentage']:.1f}}%")
            ```
            
            ## Best Practices
            
            ### Cache Configuration
            Choose appropriate cache sizes based on your application requirements:
            
            - **Small applications (< 100 contexts)**: cache_size=50
            - **Medium applications (100-1000 contexts)**: cache_size=200  
            - **Large applications (> 1000 contexts)**: cache_size=500+
            
            ### Eviction Policy Selection
            Select eviction policies based on access patterns:
            
            - **LRU**: Best for applications with temporal locality
            - **LFU**: Optimal for applications with frequency-based access
            - **TTL**: Essential for time-sensitive content
            - **Memory Pressure**: Recommended for memory-constrained environments
            
            ### Monitoring and Maintenance
            Regular monitoring ensures optimal performance:
            
            ```python
            # Schedule periodic maintenance
            import schedule
            
            def maintain_cache():
                stats = store.get_cache_stats()
                if stats['hit_rate'] < 0.8:
                    store.optimize_cache_configuration()
            
            schedule.every(1).hours.do(maintain_cache)
            ```
            """
            * random.randint(1, 2),
        },
    ]

    return workload_samples


def simulate_ai_workload(
    context_store: ContextReferenceStore, duration_seconds: int = 300
):
    """
    Simulate realistic AI workload to demonstrate dashboard capabilities.

    Args:
        context_store: The context store to populate with data
        duration_seconds: How long to run the simulation
    """
    print(" Starting AI workload simulation...")
    print(f"   Duration: {duration_seconds} seconds")
    print("   This will generate realistic context data for dashboard demonstration")

    workload_data = generate_realistic_workload_data()

    def background_workload():
        start_time = time.time()
        context_counter = 0

        while time.time() - start_time < duration_seconds:
            try:
                # Select random workload sample
                sample = random.choice(workload_data)
                content = sample["content"]
                content_type = sample["type"]

                # Add some variation to content
                if content_type == "llm_response":
                    # Simulate different query types
                    query_types = [
                        "analysis",
                        "code_review",
                        "documentation",
                        "explanation",
                    ]
                    query_type = random.choice(query_types)
                    content = f"Query Type: {query_type}\n\n{content}"

                # Store content with metadata
                context_id = context_store.store(
                    content,
                    metadata={
                        "content_type": content_type,
                        "timestamp": time.time(),
                        "simulation": True,
                        "context_number": context_counter,
                    },
                )

                context_counter += 1

                # Simulate retrieval patterns (some contexts accessed more frequently)
                if random.random() < 0.3:  # 30% chance of immediate retrieval
                    retrieved = context_store.retrieve(context_id)

                # Simulate access to older contexts (creates realistic cache patterns)
                if context_counter > 10 and random.random() < 0.2:  # 20% chance
                    try:
                        # Access a random older context (simulate realistic access patterns)
                        old_contexts = list(context_store._contexts.keys())
                        if old_contexts:
                            old_context_id = random.choice(
                                old_contexts[: min(len(old_contexts), 20)]
                            )
                            context_store.retrieve(old_context_id)
                    except:
                        pass  # Context might have been evicted

                # Variable delay to simulate realistic workload
                delay = random.uniform(
                    0.1, 2.0
                )  # 100ms to 2 seconds between operations
                time.sleep(delay)

            except Exception as e:
                print(f"   Warning: Workload simulation error: {e}")
                time.sleep(1)

        print(
            f" Workload simulation completed. Generated {context_counter} contexts."
        )

    # Start background workload
    workload_thread = threading.Thread(target=background_workload, daemon=True)
    workload_thread.start()

    return workload_thread


def main():
    """Run the TUI dashboard demonstration."""

    if not CONTEXT_STORE_AVAILABLE:
        print(" Cannot run demonstration - Context Reference Store not available")
        print(" Make sure you're in the library root directory")
        return

    if not CURSES_AVAILABLE:
        print("TUI Dashboard requires curses library")
        print("The curses library should be available on most Unix-like systems")
        return

    print("CONTEXT REFERENCE STORE - TUI DASHBOARD DEMO")
    print("=" * 55)
    print()
    print("This demonstration showcases the real-time analytics dashboard")
    print("for monitoring Context Reference Store performance.")
    print()

    # Create context store with comprehensive configuration
    print("Initializing Context Reference Store...")
    context_store = ContextReferenceStore(
        cache_size=100,
        eviction_policy=CacheEvictionPolicy.LRU,
        memory_threshold=0.8,
        enable_cache_warming=True,
        enable_compression=True,  # Enable compression for demo
        compression_min_size=500,  # Compress content > 500 bytes
        use_disk_storage=True,
        ttl_check_interval=60,  # Check TTL every minute
    )

    print("    Context store initialized with:")
    print("      • LRU cache eviction policy")
    print("      • Smart compression enabled")
    print("      • Cache warming enabled")
    print("      • Memory pressure monitoring")
    print("      • Multimodal support")

    # Pre-populate with some initial data
    print("\nPre-populating with sample data...")
    initial_data = generate_realistic_workload_data()

    for i, sample in enumerate(initial_data[:10]):  # Add first 10 samples
        context_store.store(
            sample["content"],
            metadata={"type": sample["type"], "initial_data": True, "index": i},
        )

    print(f"    Added {len(initial_data[:10])} initial contexts")

    # Start background workload simulation
    print("\nStarting background workload simulation...")
    workload_thread = simulate_ai_workload(
        context_store, duration_seconds=600
    )  # 10 minutes

    # Create and start dashboard
    print("\nLaunching TUI Dashboard...")
    print("\n" + "=" * 60)
    print("DASHBOARD CONTROLS:")
    print("   ← → : Switch between tabs (Overview, Compression, Cache, Memory, Alerts)")
    print("   ↑ ↓ : Scroll within tabs")
    print("   R   : Force refresh metrics")
    print("   C   : Clear alerts")
    print("   Q   : Quit dashboard")
    print("=" * 60)
    print()
    print("Dashboard will show:")
    print("   • Real-time performance metrics")
    print("   • Compression analytics and savings")
    print("   • Cache hit rates and efficiency")
    print("   • Memory usage and optimization tips")
    print("   • Alerts and system notifications")
    print()
    input("Press Enter to start the dashboard...")

    try:
        dashboard = create_dashboard(context_store, update_interval=1.0)
        dashboard.start()  # This will block until user quits

    except KeyboardInterrupt:
        print("\n\nDashboard stopped by user")
    except Exception as e:
        print(f"\n\nDashboard error: {e}")
    finally:
        print("\nTUI Dashboard demonstration completed!")

        # Show final statistics
        print("\nFINAL PERFORMANCE SUMMARY:")
        stats = context_store.get_cache_stats()
        print(f"   • Total contexts processed: {stats.get('total_contexts', 0)}")
        print(f"   • Final cache hit rate: {stats.get('hit_rate', 0):.1%}")
        print(f"   • Memory usage: {stats.get('memory_usage_percent', 0):.1f}%")

        # Compression summary if available
        if hasattr(context_store, "get_compression_analytics"):
            compression_stats = context_store.get_compression_analytics()
            if compression_stats.get("compression_enabled"):
                context_compression = compression_stats["context_store_stats"]
                print(
                    f"   • Contexts compressed: {context_compression.get('compressed_contexts', 0)}"
                )
                print(
                    f"   • Space savings: {context_compression.get('space_savings_percentage', 0):.1f}%"
                )
                efficiency = compression_stats["performance_impact"]
                print(
                    f"   • {efficiency.get('combined_with_reference_store', 'Efficiency data not available')}"
                )


if __name__ == "__main__":
    main()
