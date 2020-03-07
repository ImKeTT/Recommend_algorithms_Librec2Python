import math
import pandas as pd


class MetronAtK(object):
    def __init__(self, top_k):
        self._top_k = top_k
        self._subjects = None  # Subjects which we ran evaluation on

    @property
    def top_k(self):
        return self._top_k

    @top_k.setter
    def top_k(self, top_k):
        self._top_k = top_k

    @property
    def subjects(self):
        return self._subjects

    @subjects.setter
    def subjects(self, subjects):
        """
        args:
            subjects: list, [test_users, test_items, test_scores, negative users, negative items, negative scores]
        """
        assert isinstance(subjects, list)
        test_users, test_items, test_scores = subjects[0], subjects[1], subjects[2]
        neg_users, neg_items, neg_scores = subjects[3], subjects[4], subjects[5]
        # the golden set
        test = pd.DataFrame({'user': test_users,
                             'test_item': test_items,
                             'test_score': test_scores})
        # the full set
        #把 test data 和 negative data 的 ratings 混在一起（都过了一遍engine得出ratings）然后排序，计算 cal_hit_ratio & cal_ndcg
        full = pd.DataFrame({'user': neg_users + test_users,
                            'item': neg_items + test_items,
                            'score': neg_scores + test_scores})
        full = pd.merge(full, test, on=['user'], how='left')
        # rank the items according to the scores for each user
        full['rank'] = full.groupby('user')['score'].rank(method='first', ascending=False)
        full.sort_values(['user', 'rank'], inplace=True)
        self._subjects = full

    def cal_hit_ratio(self):
        """Hit Ratio @ top_K"""
        full, top_k = self._subjects, self._top_k
        top_k = full[full['rank']<=top_k]
        test_in_top_k =top_k[top_k['test_item'] == top_k['item']]  # golden items hit in the top_K items
        return len(test_in_top_k) * 1.0 / full['user'].nunique()

    def cal_ndcg(self):
        full, top_k = self._subjects, self._top_k
        #找到full(正例+负例)中rank不大于topK的对象
        top_k = full[full['rank']<=top_k]
        #计算topK中 测试对象中元素 == 全部对象中元素
        test_in_top_k =top_k[top_k['test_item'] == top_k['item']]
        #计算DCG factor
        test_in_top_k['ndcg'] = test_in_top_k['rank'].apply(lambda x: math.log(2) / math.log(1 + x)) # rank从1开始计算起
        return test_in_top_k['ndcg'].sum() * 1.0 / full['user'].nunique()
