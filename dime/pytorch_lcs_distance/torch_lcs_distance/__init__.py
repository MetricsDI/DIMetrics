import torch
import torch_lcs_distance_cuda as core


class cumulative_lcs_cuda():
    def __init__(self, gt_item, pred_item):
        """
        Computes longest common subsequence between two arrays/dictionaries of data
        Args:
            gt_item (dict/list): ground truth dict/list
            pred_item (dict/list): prediction dict/list
        """
        if type(gt_item) == dict and type(pred_item) == dict:
            self.gt_arr, self.pred_arr = self.convert_dict2arr(gt_item, pred_item)
        else:
            self.gt_arr = gt_item
            self.pred_arr = pred_item
        chars = [" "] + [str(x) for x in range(10)] + [chr(x) for x in range(ord('a'), ord('z') + 1)] + [chr(x) for x in
                                                                                                         range(ord('A'),
                                                                                                               ord('Z') + 1)]

        stop_words = [",", ".", "$", "*", "#"]
        self.encode_dict = {key: value + 1 for value, key in enumerate(chars + stop_words)}
        self.lcs_data = None

    def convert_dict2arr(self, gt_dict, pred_dict):
        gt_arr = []
        pred_arr = []
        for key in gt_dict:
            gt_arr.append(gt_dict[key])
            if key in pred_dict:
                pred_arr.append(pred_dict[key])
                del pred_dict[key]
            else:
                pred_arr.append("")

        for key in pred_dict:
            pred_arr.append(pred_dict[key])
            gt_arr.append("")

        return gt_arr, pred_arr

    def encode(self, string, max_len):
        res = []
        for char in string:
            if char in self.encode_dict:
                res.append(self.encode_dict[char])
            else:
                res.append(0)
        res += [0] * (max_len - len(res))
        return res

    def convert2tensor(self):
        gt_lens = list(map(len, self.gt_arr))
        pred_lens = list(map(len, self.pred_arr))
        max_pred_len = max(pred_lens)
        max_gt_len = max(gt_lens)
        source = list(map(lambda x: self.encode(x, max_pred_len), self.pred_arr))
        target = list(map(lambda x: self.encode(x, max_gt_len), self.gt_arr))
        source_length = torch.tensor(pred_lens, dtype=torch.int).cuda()
        target_length = torch.tensor(gt_lens, dtype=torch.int).cuda()
        source = torch.tensor(source, dtype=torch.int).cuda()
        target = torch.tensor(target, dtype=torch.int).cuda()
        return source, target, source_length, target_length

    def get_lcs_data(self):
        source, target, source_length, target_length = self.convert2tensor()
        self.lcs_data = core.lcs_distance(source, target, source_length, target_length).float()

    def get_cumlcs(self):
        if not self.lcs_data:
            self.get_lcs_data()
        tp, fp, fn = self.lcs_data[:, :].sum(dim=0)
        return tp, fp, fn

    def get_lcs(self, normalize=False):
        if not self.lcs_data:
            self.get_lcs_data()
        if normalize:
            return self.lcs_data[:, 0] / (2 * self.lcs_data[:, 0] + self.lcs_data[:, 1] + self.lcs_data[:, 2])
        else:
            return self.lcs_data[:, 0]
