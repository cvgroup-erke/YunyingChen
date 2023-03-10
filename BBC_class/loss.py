import torch
import torch.nn as nn
import torchvision.models as models


class Loss(nn.Module):
    def __init__(self, args):
        super(Loss, self).__init__()
        self.mse = nn.MSELoss()
        self.smooth_l1_loss = nn.SmoothL1Loss()
        self.l1_loss = nn.L1Loss()
        if args.cuda:
            self.mse = self.mse.cuda()
            self.smooth_l1_loss = self.smooth_l1_loss.cuda()
            self.l1_loss = self.l1_loss.cuda()

        self.perceptual_loss = False
        if args.perceptual_loss:
            self.perceptual_loss = True
            self.resnet = models.resnet34(pretrained=True).eval()
            if args.cuda:
                self.resnet = self.resnet.cuda()
            for param in self.resnet.parameters():
                param.requires_grad = False
        self.args = args

    def forward(self, pred, target, epoch_id):
        if epoch_id < self.args.epochs // 2:
            pixel_loss = self.l1_loss(pred, target)
        else:
            pixel_loss = self.mse(pred, target)
        perceptual_loss = 0
        if self.perceptual_loss:
            res_target = self.resnet(target).detach()
            res_pred = self.resnet(pred)
            perceptual_loss = self.mse(res_pred, res_target)

        return pixel_loss, perceptual_loss


class LossGAN(nn.Module):
    def __init__(self, args):
        super(LossGAN, self).__init__()
        self.mse = nn.MSELoss()
        self.smooth_l1_loss = nn.SmoothL1Loss()
        self.l1_loss = nn.L1Loss()
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.mode = args.mode
        if args.cuda:
            self.mse = self.mse.cuda()
            self.smooth_l1_loss = self.smooth_l1_loss.cuda()
            self.l1_loss = self.l1_loss.cuda()
            self.cross_entropy_loss = self.cross_entropy_loss .cuda()

        self.perceptual_loss = False
        if args.perceptual_loss:
            self.perceptual_loss = True
            self.resnet = models.resnet34(pretrained=True).eval()
            if args.cuda:
                self.resnet = self.resnet.cuda()
            for param in self.resnet.parameters():
                param.requires_grad = False
        self.args = args

    def forward(self, phase, Discriminator, img,pred, target, epoch_id):
        
        if phase == 'D':
            if self.mode =='conditional_gan' or self.mode =='flownet':
                pred = torch.cat((pred,img),dim=1)
                target = torch.cat((target,img),dim=1)
            # discriminator losses: here we use wan-gp
            # 1. adversarial loss
            adv_D_loss = - torch.mean(Discriminator(target)) + torch.mean(Discriminator(pred.detach()))

            # 2. gradient penalty
            alpha = torch.rand(target.shape[0], 1, 1, 1)
            if self.args.cuda:
                alpha = alpha.cuda(non_blocking=True)
                interpolated_x = (alpha * pred.data + (1.0 - alpha) * target.data).requires_grad_(True)
            else:
                interpolated_x = torch.FloatTensor(
                    alpha * pred.data + (1.0 - alpha) * target.data
                ).requires_grad_(True)
            out = Discriminator(interpolated_x)
            # 假设： y = f(x)
            # 则：outputs = y
            #    inputs = x
            # 求导的目的就是：dy/dx
            #
            # 若：y = f(x, m, n, d, ..., ), 则x, m, n, d都是叶子结点
            # 但，可能我们只需要(x, m), 但不需要(n, d, ...)其他的
            # 则此时，inputs就是(x, m), 意思是只对(x, m)求导
            # only_inputs: 一张图中，可能有多个“叶子结点”(如上例)
            #              若only_inputs为True，则计算图只计算inputs的梯度，而其它的叶子，就不计算了
            #
            # 其他参数：
            #     retain_graph: 保留求一次梯度之后的计算图。因为pytorch中，每一次计算梯度，计算图就会被释放。
            #                   再要做别的变量的求导，就做不了了，所以要保证做一次求导后，保留计算图。
            #                   默认情况下，它的值被与create_graph一致。也就是说，如果不单独设置retain_graph,
            #                   设置create_graph，就等于设置retain_graph。它自己本身与create_graph都默认为False
            #     create_graph: 这个是保留计算图基础上，额外建立计算图(也就等于retain_graph也为True)，可以用来算“高阶导数”
            #     grad_outputs: 这个最难理解。这个参数的设置，相当于起到了对outputs求和的作用。
            #                   为了理解这个参数，需要了解 雅阁比矩阵 [为了节省空间，下面顶头另起一行]
            # Jacobian Matrix (雅阁比矩阵)：多元方程组一次求导阵
            #  假设有f1..m(x1, x2, ..., xn) = [y1, y2, ..., ym]
            #  其中：y1 = f1(x1, x2, ..., xn)
            #       y2 = f2(x1, x2, ..., xn)
            #       ...
            #       ym = fm(x1, x2, ..., xn)
            # 则，对该方程组求一次导，其形式即为Jacobian Matrix
            #                        _                               _
            #                       |  dy1/dx1, dy1/dx2, ..., dy1/dxn |
            #  J(x1, x2, ..., xn) = |    ……       ……             ……   |
            #                       |_ dym/dx1, dym/dx2, ..., dym/dxn_|
            #  跳出当前背景，这个矩阵的一大应用，是其可获得各个x，即(x1, x2, ..., xn), 的导数。
            #  这可以用来对各个x进行迭代更新(比如用熟悉的gradient decent, 以及更复杂的levenburg-marquart方法)
            #  其在优化理论/方法中，经常见到
            #
            # 回到grad_outputs这个参数，当我们要算dy对于dx的导数时，比如要求dy/dx1,会发现
            # 此时的y是个向量(y1, y2, ..., ym),所以每个dy, 即dy1, dy2, ..., dym都要与dx1导，
            # 所以dy/dx1 = dy1/dx1 + dy2/dx1 + ... + dym/dx1
            #                               _       _
            #           = [1, 1, ..., 1] · | dy1/dx1 |
            #                 m 1s         | dy2/dx1 |
            #                              |    ……   |
            #                              |_dym/dx1_|
            # 所以[1, 1, ...., 1]这个向量，就是咱们的grad_outputs！
            # 它和out的形状应该一样，因为它的作用就是把dy/dxi的所有值给加起来，
            # 所以，最初在解释grad_outpus时我们说：“相当于起到了对outputs求和的作用”
            dxdD = torch.autograd.grad(outputs=out,
                                       inputs=interpolated_x,
                                       grad_outputs=torch.ones(out.size()).cuda(),
                                       retain_graph=True,
                                       create_graph=True,
                                       only_inputs=True)[0].view(out.shape[0], -1)
            gp_loss = torch.mean((torch.norm(dxdD, p=2) - 1) ** 2)
            return adv_D_loss, gp_loss, None
        elif phase == 'G':
            # generator losses
            # 1. adversarial loss
            if self.mode =='conditional_gan' or self.mode =='flownet':
                pred_AB = torch.cat((pred,img),dim=1)
                adv_G_loss = -torch.mean(Discriminator(pred_AB))
            else:
                adv_G_loss = -torch.mean(Discriminator(pred))

            # 2. pixel loss
            if epoch_id < self.args.epochs // 4:
                pixel_loss = self.l1_loss(pred, target)
            else:
                pixel_loss = self.mse(pred, target)

            # 3. perceptual loss
            perceptual_loss = 0
            if self.perceptual_loss:
                res_target = self.resnet(target).detach()
                res_pred = self.resnet(pred)
                perceptual_loss = self.mse(res_pred, res_target)
            return adv_G_loss, pixel_loss, perceptual_loss
        
        else:
            #Flownet loss
            # 1. pixel loss
            pixel_loss = self.mse(pred, target)
                

            # # 2. perceptual loss
            # perceptual_loss = 0
            # if self.perceptual_loss:
            #     res_target = self.resnet(target).detach()
            #     res_pred = self.resnet(pred)
            #     perceptual_loss = self.mse(res_pred, res_target)

                  
            return pixel_loss
