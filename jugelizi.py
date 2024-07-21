class aaaa():
    def __init__(self, plot_cache1=1, plot_cache2=2, pred_cache=3):
        self.plot1 = plot_cache1
        self.plot2 = plot_cache2
        self.pred = pred_cache

    def forward(self):
        return self.plot1, self.plot2, self.pred


# def chuancan():
#     b = aaaa(4, 5, 6)
#     a1, a2, a3 = b.forward()
#     print(a1, a2, a3)
#
# chuancan()