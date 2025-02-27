#升降舵通道  P:0.2; I:0.3; D:0.04
class PIDDe:
    def init_De(selfDe, KpDe, KiDe, KdDe, thetac):
        """
        初始化PID控制器

        :param KpDe: 比例增益
        :param KiDe: 积分增益
        :param KdDe: 微分增益
        :param thetac: 设定值（目标值）
        """
        selfDe.Kp = KpDe
        selfDe.Ki = KiDe
        selfDe.Kd = KdDe
        selfDe.thetac = thetac

        selfDe.previous_error_De = 0
        selfDe.integral_De = 0

    def update_De(selfDe, theta, dtDe):
        """
        更新PID控制器的输出

        :param theta: 当前俯仰角
        :param dtDe: 时间步长（时间间隔）
        :return: 升降舵
        """

        #俯仰角跟踪误差
        error_De = selfDe.thetac - theta

         # 比例项
        proportional_De = selfDe.Kp * error_De

        # 积分项
        selfDe.integral_De += error_De * dtDe
        integral_De = selfDe.Ki  * selfDe.integral_De

        # 微分项
        derivative_De = selfDe.Kd * (error_De - selfDe.previous_error_De) / dtDe

        # 更新前一次误差
        selfDe.previous_error_De = error_De

        # 计算控制输出
        output_De = proportional_De + integral_De + derivative_De

        return output_De


#滚转角通道  P:0.085; I:0.095; D:0.01
class PIDDa:
    def init_Da(selfDa, KpDa, KiDa, KdDa, phic):
        """
        初始化PID控制器

        :param KpDa: 比例增益
        :param KiDa: 积分增益
        :param KdDa: 微分增益
        :param phic: 设定值（目标值）
        """
        selfDa.Kp = KpDa
        selfDa.Ki = KiDa
        selfDa.Kd = KdDa
        selfDa.phic = phic

        selfDa.previous_error_Da = 0
        selfDa.integral_Da = 0

    def update_Da(selfDa, phi, dtDa):
        """
        更新PID控制器的输出

        :param phi: 当前滚转角
        :param dtDa: 时间步长（时间间隔）
        :return: 副翼
        """

        #滚转角跟踪误差
        error_Da = selfDa.phic - phi

         # 比例项
        proportional_Da = selfDa.Kp * error_Da

        # 积分项
        selfDa.integral_Da += error_Da * dtDa
        integral_Da = selfDa.Ki  * selfDa.integral_Da

        # 微分项
        derivative_Da = selfDa.Kd * (error_Da - selfDa.previous_error_Da) / dtDa

        # 更新前一次误差
        selfDa.previous_error_Da = error_Da

        # 计算控制输出
        output_Da = proportional_Da + integral_Da + derivative_Da

        return output_Da