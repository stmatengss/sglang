"""Scheduler lifecycle gates for the direct Mooncake radix backend."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.scheduler import Scheduler

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestSchedulerMooncakeRadixLifecycle(CustomTestCase):
    def _scheduler(self, external=True):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.enable_hierarchical_cache = False
        scheduler.enable_hicache_storage = False
        scheduler.enable_external_radix_cache = external
        scheduler.server_args = SimpleNamespace(enable_flexkv=False)
        scheduler.tree_cache = MagicMock()
        return scheduler

    def test_event_tick_drains_direct_mooncake_backend(self):
        scheduler = self._scheduler()
        scheduler._check_external_cache_events()
        scheduler.tree_cache.check_hicache_events.assert_called_once_with()

    def test_abort_releases_direct_mooncake_state(self):
        scheduler = self._scheduler()
        released = scheduler._release_external_cache_request("rid")
        self.assertTrue(released)
        scheduler.tree_cache.release_aborted_request.assert_called_once_with("rid")

    def test_default_radix_cache_does_not_call_external_lifecycle(self):
        scheduler = self._scheduler(external=False)
        scheduler._check_external_cache_events()
        released = scheduler._release_external_cache_request("rid")
        self.assertFalse(released)
        scheduler.tree_cache.check_hicache_events.assert_not_called()
        scheduler.tree_cache.release_aborted_request.assert_not_called()
