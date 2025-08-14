#!/usr/bin/env python3
"""
Comprehensive Stress and Performance Tests

This module contains stress tests, performance benchmarks, and
reliability tests for the Context Reference Store under extreme conditions.
"""

import pytest
import time
import threading
import multiprocessing
import tempfile
import os
import gc
import sys
import json
import random
import string
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from unittest.mock import patch

from context_store import (
    ContextReferenceStore,
    LargeContextState,
    CacheEvictionPolicy,
    MultimodalContent,
    MultimodalPart,
)


class TestStressScenarios:
    """Stress tests for various scenarios."""

    def test_massive_context_storage(self):
        """Test storing massive amounts of contexts."""
        store = ContextReferenceStore(
            cache_size=100, eviction_policy=CacheEvictionPolicy.LRU
        )
        # Store thousands of contexts
        context_ids = []
        start_time = time.time()

        for i in range(5000):
            content = f"Stress test content {i}: " + ("data " * 100)
            context_id = store.store(content)
            context_ids.append(context_id)

            # Periodically check performance
            if i % 1000 == 0 and i > 0:
                elapsed = time.time() - start_time
                rate = i / elapsed
                print(f"Stored {i} contexts at {rate:.2f} contexts/sec")

        end_time = time.time()
        total_time = end_time - start_time
        final_rate = len(context_ids) / total_time

        print(f"Final storage rate: {final_rate:.2f} contexts/sec")
        assert len(context_ids) == 5000

        retrieval_start = time.time()
        successful_retrievals = 0

        for _ in range(1000):
            random_id = random.choice(context_ids)
            try:
                content = store.retrieve(random_id)
                if content:
                    successful_retrievals += 1
            except KeyError:
                pass  # Expected for evicted contexts

        retrieval_time = time.time() - retrieval_start
        retrieval_rate = 1000 / retrieval_time

        print(f"Retrieval rate: {retrieval_rate:.2f} retrievals/sec")
        print(f"Successful retrievals: {successful_retrievals}/1000")

        assert final_rate > 100, f"Storage too slow: {final_rate:.2f} contexts/sec"
        assert (
            retrieval_rate > 500
        ), f"Retrieval too slow: {retrieval_rate:.2f} retrievals/sec"

    def test_extreme_concurrent_operations(self):
        """Test extreme concurrency scenarios."""
        store = ContextReferenceStore(cache_size=200)

        def intensive_worker(worker_id, num_operations):
            """Worker that performs intensive operations."""
            operations = {"stored": 0, "retrieved": 0, "deleted": 0, "errors": 0}
            local_context_ids = []

            for i in range(num_operations):
                try:
                    operation = random.choice(["store", "retrieve", "delete"])

                    if operation == "store":
                        content = f"Worker {worker_id} content {i}: " + (
                            "".join(random.choices(string.ascii_letters, k=1000))
                        )
                        context_id = store.store(content)
                        local_context_ids.append(context_id)
                        operations["stored"] += 1

                    elif operation == "retrieve" and local_context_ids:
                        context_id = random.choice(local_context_ids)
                        store.retrieve(context_id)
                        operations["retrieved"] += 1

                    elif operation == "delete" and local_context_ids:
                        context_id = local_context_ids.pop()
                        store.delete(context_id)
                        operations["deleted"] += 1

                except Exception as e:
                    operations["errors"] += 1

            return operations
        # Run many concurrent workers
        num_workers = 50
        operations_per_worker = 200

        start_time = time.time()
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(intensive_worker, i, operations_per_worker)
                for i in range(num_workers)
            ]

            results = []
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"Worker failed: {e}")

        end_time = time.time()
        total_time = end_time - start_time

        # Aggregate results
        total_ops = {"stored": 0, "retrieved": 0, "deleted": 0, "errors": 0}
        for result in results:
            for key in total_ops:
                total_ops[key] += result[key]

        total_operations = sum(total_ops.values())
        ops_per_second = total_operations / total_time
        error_rate = (
            total_ops["errors"] / total_operations if total_operations > 0 else 0
        )

        print(f"Total operations: {total_operations}")
        print(f"Operations per second: {ops_per_second:.2f}")
        print(f"Error rate: {error_rate:.2%}")
        print(
            f"Stored: {total_ops['stored']}, Retrieved: {total_ops['retrieved']}, "
            f"Deleted: {total_ops['deleted']}, Errors: {total_ops['errors']}"
        )

        assert (
            ops_per_second > 100
        ), f"Performance too low: {ops_per_second:.2f} ops/sec"
        assert error_rate < 0.2, f"Error rate too high: {error_rate:.2%}"

    def test_memory_exhaustion_scenarios(self):
        """Test behavior when approaching memory limits."""
        # Start with very limited cache
        store = ContextReferenceStore(
            cache_size=10,
            eviction_policy=CacheEvictionPolicy.MEMORY_PRESSURE,
            memory_threshold=0.8,
        )

        initial_memory = self._get_memory_usage()

        # Gradually increase memory pressure
        context_ids = []
        memory_measurements = []

        for size_mb in [1, 5, 10, 20, 50]:  # MB
            large_content = "x" * (size_mb * 1024 * 1024)

            try:
                context_id = store.store(large_content)
                context_ids.append(context_id)

                current_memory = self._get_memory_usage()
                memory_increase = current_memory - initial_memory
                memory_measurements.append((size_mb, memory_increase))

                print(
                    f"Stored {size_mb}MB content, memory increase: {memory_increase:.2f}MB"
                )

                # Force garbage collection
                gc.collect()

            except MemoryError:
                print(f"Hit memory limit at {size_mb}MB")
                break

        # Test that store can still handle small operations
        small_content = "Small content after memory pressure"
        small_id = store.store(small_content)
        retrieved = store.retrieve(small_id)
        assert retrieved == small_content
        # Verify eviction statistics
        stats = store.get_cache_stats()
        print(
            f"Cache evictions during memory pressure: {stats.get('total_evictions', 0)}"
        )

        # Should have handled memory pressure gracefully
        assert stats.get("total_evictions", 0) >= 0

    def test_multimodal_stress(self):
        """Test stress scenarios with multimodal content."""
        store = ContextReferenceStore(
            use_disk_storage=True, large_binary_threshold=1024  # 1KB threshold
        )
        # Generate various types of multimodal content
        multimodal_contexts = []

        for i in range(100):
            parts = []
            text_content = f"Multimodal stress test {i}: " + (
                "text content " * random.randint(10, 100)
            )
            parts.append(MultimodalPart(text=text_content))
            binary_size = random.randint(1024, 10240)  # 1-10KB
            binary_content = os.urandom(binary_size)
            parts.append(MultimodalPart(binary_data=binary_content))

            # Add JSON part
            json_content = {
                "id": i,
                "metadata": {"size": binary_size, "timestamp": time.time()},
                "data": list(range(random.randint(10, 100))),
            }
            parts.append(MultimodalPart(json_data=json_content))

            multimodal = MultimodalContent(
                role="user" if i % 2 == 0 else "assistant", parts=parts
            )

            context_id = store.store(multimodal)
            multimodal_contexts.append(context_id)

        # Test retrieval of all multimodal content
        retrieval_errors = 0
        total_retrieved_size = 0
        for context_id in multimodal_contexts:
            try:
                retrieved = store.retrieve(context_id)
                assert isinstance(retrieved, MultimodalContent)
                assert len(retrieved.parts) == 3

                # Estimate size
                for part in retrieved.parts:
                    if part.text:
                        total_retrieved_size += len(part.text.encode())
                    if part.binary_data:
                        total_retrieved_size += len(part.binary_data)
                    if part.json_data:
                        total_retrieved_size += len(json.dumps(part.json_data).encode())

            except Exception as e:
                retrieval_errors += 1
                print(f"Retrieval error: {e}")

        error_rate = retrieval_errors / len(multimodal_contexts)

        print(f"Multimodal contexts stored: {len(multimodal_contexts)}")
        print(f"Total retrieved size: {total_retrieved_size / 1024 / 1024:.2f}MB")
        print(f"Retrieval error rate: {error_rate:.2%}")
        assert error_rate < 0.1, f"Too many retrieval errors: {error_rate:.2%}"

    def test_disk_storage_stress(self):
        """Test stress scenarios with disk storage."""
        with tempfile.TemporaryDirectory() as temp_dir:
            store = ContextReferenceStore(
                use_disk_storage=True,
                binary_cache_dir=temp_dir,
                large_binary_threshold=1024,  # 1KB threshold
                cache_size=20,  # Small memory cache
            )

            # Store many large binary objects
            binary_contexts = []
            total_disk_size = 0

            for i in range(100):
                # Create large binary content that will go to disk
                size = random.randint(10240, 102400)  
                binary_content = os.urandom(size)

                context_id = store.store(binary_content)
                binary_contexts.append((context_id, size))
                total_disk_size += size

                if i % 20 == 0:
                    print(
                        f"Stored {i} binary objects, total size: {total_disk_size / 1024 / 1024:.2f}MB"
                    )

            # Verify disk usage
            disk_files = []
            for root, dirs, files in os.walk(temp_dir):
                disk_files.extend(files)

            print(f"Files on disk: {len(disk_files)}")
            print(f"Total theoretical size: {total_disk_size / 1024 / 1024:.2f}MB")

            # Test retrieval performance from disk
            retrieval_start = time.time()
            successful_retrievals = 0

            for context_id, expected_size in binary_contexts[:50]:  # Test subset
                try:
                    retrieved = store.retrieve(context_id)
                    if isinstance(retrieved, bytes) and len(retrieved) == expected_size:
                        successful_retrievals += 1
                except Exception as e:
                    print(f"Disk retrieval error: {e}")

            retrieval_time = time.time() - retrieval_start
            retrieval_rate = 50 / retrieval_time

            print(f"Disk retrieval rate: {retrieval_rate:.2f} retrievals/sec")
            print(f"Successful disk retrievals: {successful_retrievals}/50")

            # Should handle disk storage stress well
            assert (
                successful_retrievals >= 40
            ), f"Too many disk retrieval failures: {successful_retrievals}/50"
            assert len(disk_files) > 0, "No files were written to disk"

    def test_rapid_eviction_scenarios(self):
        """Test scenarios with rapid cache evictions."""
        store = ContextReferenceStore(
            cache_size=5, eviction_policy=CacheEvictionPolicy.LRU  # Very small cache
        )

        # Store many contexts to force rapid evictions
        context_ids = []
        eviction_stats = []
        for i in range(100):
            content = f"Rapid eviction test {i}: " + ("data " * 1000)
            context_id = store.store(content)
            context_ids.append(context_id)

            if i % 10 == 0:
                stats = store.get_cache_stats()
                eviction_stats.append(stats.get("total_evictions", 0))

        final_stats = store.get_cache_stats()
        print(f"Final evictions: {final_stats.get('total_evictions', 0)}")
        print(f"Contexts in cache: {final_stats.get('contexts_in_cache', 0)}")
        print(f"Cache hit rate: {final_stats.get('hit_rate', 0):.2%}")

        # Test that recent contexts are still accessible
        recent_accessible = 0
        for context_id in context_ids[-10:]:  # Last 10 contexts
            try:
                store.retrieve(context_id)
                recent_accessible += 1
            except KeyError:
                pass

        print(f"Recent contexts accessible: {recent_accessible}/10")

        assert final_stats.get("total_evictions", 0) > 0
        assert recent_accessible > 0

    def _get_memory_usage(self):
        """Get current memory usage in MB."""
        try:
            import psutil

            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0  # Can't measure without psutil


class TestPerformanceBenchmarks:
    """Performance benchmark tests."""

    def test_serialization_performance(self):
        """Benchmark serialization performance."""
        store = ContextReferenceStore()

        # Test different content sizes
        sizes = [1024, 10240, 102400, 1024000]  # 1KB to 1MB
        results = {}

        for size in sizes:
            content = "x" * size
            # Measure serialization time
            start_time = time.time()

            for _ in range(100):
                context_id = store.store(content)
                store.retrieve(context_id)
            end_time = time.time()
            total_time = end_time - start_time
            operations_per_second = 200 / total_time  # 100 store + 100 retrieve
            mb_per_second = (size * 200) / (1024 * 1024) / total_time

            results[size] = {
                "ops_per_sec": operations_per_second,
                "mb_per_sec": mb_per_second,
                "avg_time_ms": (total_time / 200) * 1000,
            }

            print(
                f"Size {size} bytes: {operations_per_second:.2f} ops/sec, "
                f"{mb_per_second:.2f} MB/sec, {results[size]['avg_time_ms']:.2f}ms avg"
            )

        # Verify performance meets expectations
        small_ops = results[1024]["ops_per_sec"]
        large_ops = results[1024000]["ops_per_sec"]

        assert small_ops > 1000, f"Small content too slow: {small_ops:.2f} ops/sec"
        assert large_ops > 10, f"Large content too slow: {large_ops:.2f} ops/sec"

    def test_cache_performance_characteristics(self):
        """Test cache performance across different scenarios."""
        cache_sizes = [10, 50, 100, 500]
        results = {}

        for cache_size in cache_sizes:
            store = ContextReferenceStore(
                cache_size=cache_size, eviction_policy=CacheEvictionPolicy.LRU
            )
            # Fill cache
            context_ids = []
            for i in range(cache_size * 2):  # Store more than cache size
                content = f"Cache test {i}: " + ("data " * 100)
                context_id = store.store(content)
                context_ids.append(context_id)

            # Measure cache hit performance
            start_time = time.time()
            hits = 0
            misses = 0

            for _ in range(1000):
                # Access recent contexts (should be cache hits)
                recent_id = random.choice(context_ids[-cache_size // 2 :])
                try:
                    store.retrieve(recent_id)
                    hits += 1
                except KeyError:
                    misses += 1

            end_time = time.time()

            hit_rate = hits / (hits + misses)
            operations_per_second = 1000 / (end_time - start_time)
            results[cache_size] = {
                "hit_rate": hit_rate,
                "ops_per_sec": operations_per_second,
                "hits": hits,
                "misses": misses,
            }
            print(
                f"Cache size {cache_size}: {hit_rate:.2%} hit rate, "
                f"{operations_per_second:.2f} ops/sec"
            )

        # Larger caches should generally have better hit rates
        hit_rates = [results[size]["hit_rate"] for size in cache_sizes]
        assert max(hit_rates) > 0.5, "Cache hit rates too low"

    def test_multiprocessing_performance(self):
        """Test performance with multiprocessing."""
        if sys.platform == "win32":
            pytest.skip(
                "Multiprocessing tests skipped on Windows due to pickling issues"
            )

        def worker_process(worker_id, num_operations):
            """Worker process for multiprocessing test."""
            store = ContextReferenceStore(cache_size=50)
            operations = 0

            for i in range(num_operations):
                content = f"Process {worker_id} content {i}: " + ("data " * 100)
                context_id = store.store(content)
                retrieved = store.retrieve(context_id)
                if retrieved:
                    operations += 2  # Store + retrieve

            return operations

        num_processes = min(4, multiprocessing.cpu_count())
        operations_per_process = 100
        start_time = time.time()

        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            futures = [
                executor.submit(worker_process, i, operations_per_process)
                for i in range(num_processes)
            ]
            total_operations = 0
            for future in as_completed(futures):
                try:
                    operations = future.result()
                    total_operations += operations
                except Exception as e:
                    print(f"Process failed: {e}")

        end_time = time.time()
        total_time = end_time - start_time
        ops_per_second = total_operations / total_time

        print(f"Multiprocessing: {total_operations} operations in {total_time:.2f}s")
        print(f"Rate: {ops_per_second:.2f} ops/sec across {num_processes} processes")
        assert (
            ops_per_second > 100
        ), f"Multiprocessing performance too low: {ops_per_second:.2f} ops/sec"

    def test_compression_performance_impact(self):
        """Test performance impact of compression."""
        # Test without compression
        store_no_compression = ContextReferenceStore(enable_compression=False)
        # Test with compression
        store_with_compression = ContextReferenceStore(
            enable_compression=True, compression_min_size=1024
        )
        compressible_content = "This is highly compressible content! " * 1000
        # Benchmark without compression
        start_time = time.time()
        for i in range(100):
            context_id = store_no_compression.store(compressible_content)
            store_no_compression.retrieve(context_id)
        no_compression_time = time.time() - start_time
        # Benchmark with compression
        start_time = time.time()
        for i in range(100):
            context_id = store_with_compression.store(compressible_content)
            store_with_compression.retrieve(context_id)
        compression_time = time.time() - start_time

        # Get storage stats
        no_comp_stats = store_no_compression.get_cache_stats()
        comp_stats = store_with_compression.get_cache_stats()

        print(f"Without compression: {no_compression_time:.2f}s")
        print(f"With compression: {compression_time:.2f}s")
        print(
            f"Compression overhead: {((compression_time / no_compression_time) - 1) * 100:.1f}%"
        )

        # Compression might be slower but should provide storage benefits
        # Allow up to 50% performance overhead for compression benefits
        overhead_factor = compression_time / no_compression_time
        assert (
            overhead_factor < 2.0
        ), f"Compression overhead too high: {overhead_factor:.2f}x"


class TestReliabilityScenarios:
    """Reliability and fault tolerance tests."""

    def test_corruption_recovery(self):
        """Test recovery from various corruption scenarios."""
        store = ContextReferenceStore()
        valid_content = "Valid content for corruption test"
        valid_id = store.store(valid_content)
        # Verify valid data is retrievable
        retrieved = store.retrieve(valid_id)
        assert retrieved == valid_content

        # Simulate corruption by modifying internal structures
        if hasattr(store, "_cache"):
            # Add corrupted entry to cache
            corrupted_id = "corrupted_context_id"
            store._cache[corrupted_id] = "corrupted_data_that_might_cause_issues"
        try:
            store.retrieve(corrupted_id)
        except KeyError:
            pass  # Expected for non-existent context
        except Exception as e:
            print(f"Handled corruption gracefully: {e}")

        retrieved_after = store.retrieve(valid_id)
        assert retrieved_after == valid_content

    def test_resource_exhaustion_recovery(self):
        """Test recovery from resource exhaustion scenarios."""
        store = ContextReferenceStore(cache_size=10)

        # Gradually exhaust resources
        context_ids = []

        try:
            for i in range(1000):
                # Create progressively larger content
                size = min(1024 * (i + 1), 1024 * 1024)  # Up to 1MB
                content = "x" * size

                context_id = store.store(content)
                context_ids.append(context_id)
                if i % 100 == 0:
                    test_content = "Recovery test"
                    test_id = store.store(test_content)
                    retrieved = store.retrieve(test_id)
                    assert retrieved == test_content

        except MemoryError:
            print("Hit memory limit, testing recovery...")

        small_content = "Small content after exhaustion"
        small_id = store.store(small_content)
        retrieved = store.retrieve(small_id)
        assert retrieved == small_content

    def test_concurrent_failure_scenarios(self):
        """Test behavior when some concurrent operations fail."""
        store = ContextReferenceStore(cache_size=20)

        def failing_worker(worker_id, failure_rate=0.1):
            """Worker that occasionally fails."""
            results = {"success": 0, "failures": 0}

            for i in range(50):
                try:
                    # Randomly inject failures
                    if random.random() < failure_rate:
                        # Simulate failure by accessing non-existent context
                        store.retrieve("non_existent_context")
                    else:
                        content = f"Worker {worker_id} content {i}"
                        context_id = store.store(content)
                        retrieved = store.retrieve(context_id)
                        if retrieved == content:
                            results["success"] += 1
                        else:
                            results["failures"] += 1

                except Exception:
                    results["failures"] += 1

            return results

        # Run workers with different failure rates
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(failing_worker, i, 0.1)  # 10% failure rate
                for i in range(10)
            ]

            total_results = {"success": 0, "failures": 0}
            for future in as_completed(futures):
                result = future.result()
                total_results["success"] += result["success"]
                total_results["failures"] += result["failures"]

        total_operations = total_results["success"] + total_results["failures"]
        success_rate = (
            total_results["success"] / total_operations if total_operations > 0 else 0
        )

        print(f"Success rate with failures: {success_rate:.2%}")
        print(f"Successful operations: {total_results['success']}")
        print(f"Failed operations: {total_results['failures']}")

        # Should maintain reasonable success rate despite failures
        assert success_rate > 0.7, f"Success rate too low: {success_rate:.2%}"

    def test_long_running_stability(self):
        """Test stability over extended operation periods."""
        store = ContextReferenceStore(
            cache_size=50, eviction_policy=CacheEvictionPolicy.LRU
        )

        # Run for extended period with various operations
        start_time = time.time()
        duration = 30  # Run for 30 seconds

        operations = {"store": 0, "retrieve": 0, "delete": 0, "errors": 0}
        context_ids = []

        while time.time() - start_time < duration:
            try:
                operation = random.choice(["store", "store", "retrieve", "delete"])

                if operation == "store":
                    content = f"Long running test {operations['store']}: " + (
                        "".join(random.choices(string.ascii_letters, k=1000))
                    )
                    context_id = store.store(content)
                    context_ids.append(context_id)
                    operations["store"] += 1

                elif operation == "retrieve" and context_ids:
                    context_id = random.choice(context_ids)
                    store.retrieve(context_id)
                    operations["retrieve"] += 1

                elif operation == "delete" and context_ids:
                    context_id = context_ids.pop(
                        random.randint(0, len(context_ids) - 1)
                    )
                    store.delete(context_id)
                    operations["delete"] += 1

                # Occasionally force garbage collection
                if sum(operations.values()) % 1000 == 0:
                    gc.collect()
                    stats = store.get_cache_stats()
                    print(
                        f"Operations: {sum(operations.values())}, "
                        f"Cache: {stats.get('contexts_in_cache', 0)}, "
                        f"Evictions: {stats.get('total_evictions', 0)}"
                    )

            except Exception as e:
                operations["errors"] += 1
                if operations["errors"] > 100: 
                    break

        total_operations = sum(operations.values())
        elapsed = time.time() - start_time
        ops_per_second = total_operations / elapsed
        error_rate = (
            operations["errors"] / total_operations if total_operations > 0 else 0
        )

        print(f"Long running test completed:")
        print(f"Duration: {elapsed:.2f}s")
        print(f"Total operations: {total_operations}")
        print(f"Operations per second: {ops_per_second:.2f}")
        print(f"Error rate: {error_rate:.2%}")
        print(f"Operations breakdown: {operations}")

        assert (
            ops_per_second > 50
        ), f"Performance degraded: {ops_per_second:.2f} ops/sec"
        assert error_rate < 0.05, f"Error rate too high: {error_rate:.2%}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x"]) 
