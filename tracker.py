import numpy as np

VELOCITY_BUFFER = 5


def iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-6)


def iou_cost_matrix(tracks, dets):
    cost = np.zeros((len(tracks), len(dets)), dtype=np.float32)
    for i, t in enumerate(tracks):
        tb = t.predict()
        last = t.bbox
        for j, d in enumerate(dets):
            cost[i, j] = 1.0 - max(iou(tb, d['bbox']), iou(last, d['bbox']))
    return cost


def greedy_match(cost, max_cost):
    matched_t = set()
    matched_d = set()
    pairs = []
    rows, cols = cost.shape
    for _ in range(min(rows, cols)):
        idx = np.unravel_index(np.argmin(cost), cost.shape)
        if cost[idx] > max_cost:
            break
        t, d = idx
        matched_t.add(t)
        matched_d.add(d)
        pairs.append((t, d))
        cost[t, :] = 1.0
        cost[:, d] = 1.0
    return matched_t, matched_d, pairs


class TrackedObject:
    def __init__(self, track_id, bbox, label, confidence, activated=True):
        self.track_id = track_id
        self.label = label
        self.confidence = confidence
        self.activated = activated
        self.bboxes = [bbox]
        self.centroids = [((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)]
        self.disappeared = 0
        self.age = 0
        self.last_crossing = None
        self.last_crossing_frame = -60

    @property
    def centroid(self):
        return self.centroids[-1]

    @property
    def bbox(self):
        return self.bboxes[-1]

    @property
    def prev_centroid(self):
        return self.centroids[-2] if len(self.centroids) >= 2 else None

    def predict(self):
        if len(self.centroids) >= 3:
            dx = int(np.median([self.centroids[i][0] - self.centroids[i - 1][0]
                                for i in range(-2, 0)]))
            dy = int(np.median([self.centroids[i][1] - self.centroids[i - 1][1]
                                for i in range(-2, 0)]))
        elif len(self.centroids) == 2:
            dx = self.centroids[-1][0] - self.centroids[-2][0]
            dy = self.centroids[-1][1] - self.centroids[-2][1]
        else:
            dx, dy = 0, 0
        x1, y1, x2, y2 = self.bboxes[-1]
        return (x1 + dx, y1 + dy, x2 + dx, y2 + dy)

    def update(self, bbox, confidence):
        self.bboxes.append(bbox)
        cx = (bbox[0] + bbox[2]) // 2
        cy = (bbox[1] + bbox[3]) // 2
        self.centroids.append((cx, cy))
        if len(self.bboxes) > VELOCITY_BUFFER:
            self.bboxes.pop(0)
            self.centroids.pop(0)
        self.confidence = confidence
        self.age += 1
        self.disappeared = 0


class ByteTrack:
    def __init__(self, conf_thresh=0.5, max_lost=30, iou_thresh=0.15, second_iou_thresh=0.1):
        self.next_id = 0
        self.objects = {}
        self.conf_thresh = conf_thresh
        self.max_lost = max_lost
        self.iou_thresh = iou_thresh
        self.second_iou_thresh = second_iou_thresh

    def update(self, detections):
        if not detections:
            died = []
            for tid, obj in self.objects.items():
                obj.disappeared += 1
                if obj.disappeared > self.max_lost:
                    died.append(tid)
            for tid in died:
                del self.objects[tid]
            return [o for o in self.objects.values() if o.activated]

        high = [d for d in detections if d['confidence'] >= self.conf_thresh]
        low = [d for d in detections if d['confidence'] < self.conf_thresh]

        active = {tid: obj for tid, obj in self.objects.items() if obj.activated}
        lost = {tid: obj for tid, obj in self.objects.items() if not obj.activated}

        if active and high:
            t_list = list(active.values())
            t_ids = list(active.keys())
            cost = iou_cost_matrix(t_list, high)
            matched_t, matched_d, pairs = greedy_match(cost, 1.0 - self.iou_thresh)
            for t_idx, d_idx in pairs:
                tid = t_ids[t_idx]
                active[tid].update(high[d_idx]['bbox'], high[d_idx]['confidence'])
            unmatched_active = [t_ids[i] for i in range(len(t_ids)) if i not in matched_t]
            unmatched_high = [high[i] for i in range(len(high)) if i not in matched_d]
        else:
            unmatched_active = list(active.keys())
            unmatched_high = high[:]

        if lost and unmatched_high:
            t_list = list(lost.values())
            t_ids = list(lost.keys())
            cost = iou_cost_matrix(t_list, unmatched_high)
            _, matched_d, pairs = greedy_match(cost, 1.0 - self.second_iou_thresh)
            for t_idx, d_idx in pairs:
                tid = t_ids[t_idx]
                lost[tid].update(unmatched_high[d_idx]['bbox'], unmatched_high[d_idx]['confidence'])
                lost[tid].activated = True
                active[tid] = lost[tid]
                del lost[tid]
                unmatched_high[d_idx] = None
            unmatched_high = [d for d in unmatched_high if d is not None]

        for tid in unmatched_active:
            self.objects[tid].disappeared += 1
            if self.objects[tid].disappeared > self.max_lost:
                del self.objects[tid]

        remaining_active = {tid: o for tid, o in active.items()
                            if tid in self.objects and self.objects[tid].activated}
        if remaining_active and low:
            t_list = list(remaining_active.values())
            t_ids = list(remaining_active.keys())
            cost = iou_cost_matrix(t_list, low)
            _, matched_d, pairs = greedy_match(cost, 1.0 - self.second_iou_thresh)
            for t_idx, d_idx in pairs:
                tid = t_ids[t_idx]
                remaining_active[tid].update(low[d_idx]['bbox'], low[d_idx]['confidence'])
                low[d_idx] = None
            low = [d for d in low if d is not None]

        for d in unmatched_high + low:
            if d is None:
                continue
            tid = self.next_id
            self.next_id += 1
            self.objects[tid] = TrackedObject(tid, d['bbox'], d['label'], d['confidence'])

        return [o for o in self.objects.values() if o.activated]
