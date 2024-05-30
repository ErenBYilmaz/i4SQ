import gc
from typing import Optional, Set


from lib.print_exc_plus import type_string, value_to_string
from lib.util import EBC

def default_exclusions(o) -> bool:
    if isinstance(o, MemoryLeakDebugger):
        return True
    if 'pydevd' in str(type(o)):
        return True
    if 'MemoryLeakDebugger' in str(type(o)):
        return True
    if isinstance(o, (set, list, tuple, dict)) and len(o) == 0:
        return True

class MemoryLeakDebugger(EBC):
    def __init__(self, obj_ids_before: Optional[Set[int]] = None, obj_ids_after: Optional[Set[int]] = None):
        self.obj_ids_before = obj_ids_before
        self.obj_ids_after = obj_ids_after

    def start(self):
        self.obj_ids_before = set(id(o) for o in gc.get_objects())

    def stop(self):
        gc.collect()
        self.obj_ids_after = set(id(o) for o in gc.get_objects())

    def new_objects(self, ignore=default_exclusions):
        gc.collect()
        return [o for o in gc.get_objects()
                if id(o) not in self.obj_ids_before
                if o is not self.obj_ids_before
                if o is not self
                if o is not self.obj_ids_after
                if not ignore(o)]

    def print_referrer_tree_and_return_leaves(self, root, max_depth=3, max_children=10, indent=0, exclude_old_objs=True):
        prefix = indent * ' '
        if len(prefix) >= 2:
            prefix = prefix[:-2] + '└─'
        if max_depth <= 0:
            print(prefix + '...')
            return [root]
        existed_before = '(existed before) ' if id(root) in self.obj_ids_before else ''
        print(prefix + existed_before + type_string(root) + ' : ' + value_to_string(root)[:200])
        referrers = [referrer for referrer in gc.get_referrers(root)
                     if (not exclude_old_objs) or id(referrer) not in self.obj_ids_before
                     if id(referrer) in self.obj_ids_after][:max_children]
        leaves = []
        for referrer_idx, referrer in enumerate(referrers):
            leaves += self.print_referrer_tree_and_return_leaves(referrer, max_depth - 1, indent=indent + 2, exclude_old_objs=exclude_old_objs)
            if referrer_idx + 1 == max_children:
                print(prefix + '...')
                break
        return leaves
# m = MemoryLeakDebugger(obj_ids_before, obj_ids_after)
# o = m.new_objects()[20]
