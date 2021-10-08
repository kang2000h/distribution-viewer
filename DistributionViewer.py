import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

class DistributionViewer():
    def __init__(self, feature_list, label_list=None, title=None, picker=5, color_list="bgrcmykw"):
        self._fig = None
        self._plot_line_list = []
        self._label_list_per_plot_line = []

        self._init_figure(feature_list, label_list, title, picker, color_list)
        return

    def _init_figure(self, feature_list, label_list=None, title=None, picker=5, color_list="bgrcmykw"):
        self._fig = plt.figure()
        ax = self._fig.add_subplot(111)

        self._feature_list = np.array(feature_list)
        if title :
            ax.set_title(title)

        if label_list is not None:
            self._plot_line_list = []
            self._label_list_per_plot_line = []
            for ind, c_label in enumerate(np.unique(label_list)):
                self._line = ax.plot(feature_list[label_list == c_label, 0], feature_list[label_list == c_label, 1], 'o',
                                     picker=picker, color=color_list[ind], label=c_label)
                ax.legend()
                # self._line = ax.scatter(feature_list[label_list == c_label, 0], feature_list[label_list == c_label, 1], marker='o',
                #                       picker=picker, color=color_list[ind])

                self._plot_line_list.append(self._line[0])
                self._label_list_per_plot_line.append(np.where(label_list == ind))
        else :
            self._line = ax.plot(feature_list[:0], feature_list[:1], 'o', picker=picker)
            self._line = self._line[0]
        return

    # def show_labeled_feature_dist(self, x_list, y_list):
    #     return

    def show_2d_features_dist(self, origin_X, type, cmap=None, **infos):

        #print("origin_X".shape)
        onpick = self._get_onclick(origin_X=origin_X, type=type, cmap=cmap, infos=infos)
        self._fig.canvas.mpl_connect('pick_event', onpick)
        plt.show()
        return

    def _get_var_type(self, var):
        # print("[!], type(var) in _get_var_type", var, type(var))
        if isinstance(var, int) or isinstance(var, np.int32):
            return 'd'
        elif isinstance(var, float) or isinstance(var, np.float64):
            return 'f'
        elif isinstance(var, str):
            return 's'
        else:
            return 'f'

    def _create_strfmt(self, size_gap, len_var, var_list, align_l=True, var_name_list=None):
        str_fmt = ""  # '%-10s%-10s%-10s\n'

        for ind in range(len_var):
            if var_name_list :
                str_fmt+=var_name_list[ind]+':'
            str_fmt += '%'
            if align_l:
                str_fmt += '-'
            type_str = self._get_var_type(var_list[ind])
            try:
                str_fmt += str(size_gap) + type_str
            except TypeError as te:
                print(te)
                print("size_gap", size_gap, "type_str", type_str)
            str_fmt += '\n'

        return str_fmt

    def _get_onclick(self, origin_X, type='plot', cmap=None, **infos):
        def onpick(event):

            var_dict = infos['infos']
            # print("event", event) # <matplotlib.backend_bases.PickEvent object at 0x0000019FE5D80828>
            # print("event.artist", event.artist) # Line2D(_line0)
            # print("event.ind", event.ind)
            #if event.artist != self._line: return True
            if event.artist not in self._plot_line_list : return True
            else :
                line_ind = self._plot_line_list.index(event.artist)
                event_label_list = np.array(self._label_list_per_plot_line[line_ind])[0]
            real_point_ind = event_label_list[event.ind]

            N = len(event.ind)

            if not N: return True

            figi = plt.figure()
            for subplotnum, dataind in enumerate(real_point_ind):
                #figi.subplots_adjust(left=0.1, right=0.8, top=0.9, bottom=0.1)
                # ax1 = figi.add_subplot(N, 1, subplotnum + 1)

                # for origin_X
                if origin_X is not None:
                    grid = gridspec.GridSpec(N, 2, width_ratios=[2, 1])
                    ax1 = figi.add_subplot(grid[subplotnum, 0])

                    if type=='plot':
                        ax1.plot(origin_X[dataind])
                    elif type=='imshow_2d':
                        ax1.imshow(origin_X[dataind], cmap=cmap)
                    elif type=='imshow_3d': # (D, H, W)
                        ax1.imshow(origin_X[dataind], cmap=cmap)

                field_to_write = sorted(var_dict.items(), key=lambda x: x[0])
                var_name_list = ['x', 'y']+list(map(lambda x: x[0], field_to_write))
                var_list = (self._feature_list[dataind][0], self._feature_list[dataind][1])+tuple(map(lambda x: x[1][dataind], field_to_write))
                _strfmt = self._create_strfmt(8, len(var_list), var_list, align_l=True, var_name_list=var_name_list)

                # for variables (text) related to origin X
                if origin_X is not None:
                    ax2 = figi.add_subplot(grid[subplotnum, 1])
                else :
                    ax2 = figi.add_subplot(N, 1, 1)
                # ax.text(0.05, 0.9, _strfmt % var_list,
                #         transform=ax.transAxes, va='top')

                ax2.text(0.05, 0.9, _strfmt % var_list, va='top')
                ax2.get_xaxis().set_visible(False)
                ax2.get_yaxis().set_visible(False)
                #ax1.set_ylim(-0.5, 1.5)

                #print(xs[dataind], ys[dataind], label_list[dataind], dataind)
            figi.show()
            #figi.close()

            return True
        return onpick




if __name__ == '__main__':
    mode = 'test' # 'fbb_test' # 'test' # 'dev'
    if mode == 'dev':
        print("mode:", mode)
        from tensorflow.keras import utils
        # set input
        xs = np.array(list(range(10)))
        ys = np.array([1]*10)
        label_list = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1])
        #label_list = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1])
        print(label_list)

        #feature_X = np.concatenate([xs[:,None], ys[:,None]], axis=1)
        #print("feature_X.shape", feature_X.shape)

        fig = plt.figure()
        ax = fig.add_subplot(111)

        ax.set_title('click on point to plot time series')
        color_list="rbgk"

        plot_line_list = []
        label_list_per_plot_line = []
        for ind, clabel in enumerate(np.unique(label_list)):
            print("ind", ind, clabel)

            line, = ax.plot(xs[label_list==ind], ys[label_list==ind], 'o', picker=5, color=color_list[ind])
            plot_line_list.append(line)
            label_list_per_plot_line.append(np.where(label_list==ind))
        #line, = ax.plot(xs, ys, 'o', picker=5)

        def onpick(event):
            print("event", event) # <matplotlib.backend_bases.PickEvent object at 0x0000019FE5D80828>
            print("event.artist", event.artist) # Line2D(_line0)
            print("event.ind", event.ind)
            #if event.artist != line1: return True
            if event.artist not in plot_line_list : return True
            else :
                line_ind = plot_line_list.index(event.artist)
                event_label_list = np.array(label_list_per_plot_line[line_ind])[0]
            real_point_ind = event_label_list[event.ind]
            N = len(event.ind)
            print("N", N)
            if not N: return True

            figi = plt.figure()
            for subplotnum, dataind in enumerate(event.ind):
                figi.subplots_adjust(left=0.1, right=0.8, top=0.9, bottom=0.1)
                ax2 = figi.add_subplot(N, 1, subplotnum + 1)

                ax2.plot(origin_X[real_point_ind])
                # ax2.text(0.05, 0.9, 'mu=%1.3f\nsigma=%1.3f' % (xs[dataind], ys[dataind]),
                #         transform=ax2.transAxes, va='top')
                #figi.text(0.05, 0.99, 'mu=%1.3f\nsigma=%1.3f' % (xs[dataind], ys[dataind]),  va='top')
                # figi.text(0.82, 0.90, 'mu=%1.3f\nsigma=%1.3f\nlabel=%d' % (xs[dataind], ys[dataind], label_list[dataind]),
                #           transform=ax2.transAxes, va='top')
                figi.text(0.82, 0.90,
                          'mu=%1.3f\nsigma=%1.3f\nlabel=%d' % (xs[real_point_ind], ys[real_point_ind], label_list[real_point_ind]),
                          transform=ax2.transAxes, va='top')
                ax2.set_ylim(-0.5, 1.5)
                print(xs[real_point_ind], ys[real_point_ind], label_list[real_point_ind])
            figi.show()

            return True


        fig.canvas.mpl_connect('pick_event', onpick)
        plt.show()
        plt.close()
    elif mode == 'test':
        print("mode", mode)

        # set input
        origin_X = np.random.rand(100, 1000)
        #print("origin_X.shape", origin_X.shape)
        #print("origin_X[0].shape", origin_X[0].shape)
        print("X.shape", origin_X.shape)

        # feature
        xs = np.mean(origin_X, axis=1)
        ys = np.std(origin_X, axis=1)
        feature_list = np.concatenate([xs[:,None], ys[:,None]], axis=1)
        label_list = np.array([0, 1]*50)

        # parameters
        title = "feature distribution viewer test"
        picker = 5
        color_list = "rbgk"

        dis_view = DistributionViewer(feature_list, label_list, title, picker, color_list)
        dis_view.show_2d_features_dist(origin_X=origin_X, type='plot', mu=xs.astype(np.str), std=ys.astype(np.str))
