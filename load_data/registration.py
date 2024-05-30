import SimpleITK

from lib.main_wrapper import main_wrapper
from lib.util import EBC


class RegistrationOptimizer(EBC):
    def set_for_registration(self, registration: SimpleITK.ImageRegistrationMethod):
        raise NotImplementedError


class RegularStepGradientDescent(RegistrationOptimizer):
    # noinspection PyPep8Naming
    def __init__(self,
                 learningRate: float,
                 minStep: float,
                 numberOfIterations: int,
                 relaxationFactor=0.5,
                 gradientMagnitudeTolerance=1e-4, ):
        self.learningRate = learningRate
        self.minStep = minStep
        self.numberOfIterations = numberOfIterations
        self.relaxationFactor = relaxationFactor
        self.gradientMagnitudeTolerance = gradientMagnitudeTolerance

    def set_for_registration(self, registration: SimpleITK.ImageRegistrationMethod):
        registration.SetOptimizerAsRegularStepGradientDescent(self.learningRate, self.minStep, self.numberOfIterations, self.relaxationFactor, self.gradientMagnitudeTolerance)


class GradientDescentLineSearch(RegistrationOptimizer):
    # noinspection PyPep8Naming
    def __init__(self,
                 learningRate: float,
                 numberOfIterations: int,
                 convergenceMinimumValue: float,
                 relaxationFactor=0.5,
                 convergenceWindowSize=5, ):
        self.learningRate = learningRate
        self.convergenceMinimumValue = convergenceMinimumValue
        self.numberOfIterations = numberOfIterations
        self.relaxationFactor = relaxationFactor
        self.convergenceWindowSize = convergenceWindowSize

    def set_for_registration(self, registration: SimpleITK.ImageRegistrationMethod):
        registration.SetOptimizerAsGradientDescentLineSearch(self.learningRate, self.numberOfIterations, self.convergenceMinimumValue, self.convergenceWindowSize)


class VertebraRegistration(EBC):
    """
    For available transformations, metrics, optimizers, and interpolators see https://simpleitk.readthedocs.io/en/master/registrationOverview.html
    Also quite helpful: https://simpleitk.readthedocs.io/en/master/link_ImageRegistrationMethod1_docs.html
    """

    def __init__(self,
                 transformation=SimpleITK.VersorTransform(),
                 metric=SimpleITK.ImageRegistrationMethod.SetMetricAsMeanSquares,
                 optimizer=GradientDescentLineSearch(learningRate=1., numberOfIterations=200, convergenceMinimumValue=1e-5, convergenceWindowSize=5),
                 # optimizer=RegularStepGradientDescent(0.004, 0.0001, 200),
                 interpolator=SimpleITK.sitkLinear,
                 verbose=False):
        self.transformation = transformation
        self.metric = metric
        self.optimizer = optimizer
        self.interpolator = interpolator

        self.sitk_registration = SimpleITK.ImageRegistrationMethod()
        self.metric(self.sitk_registration)
        self.optimizer.set_for_registration(self.sitk_registration)
        self.sitk_registration.SetInitialTransform(self.transformation)
        self.sitk_registration.SetInterpolator(self.interpolator)
        # self.sitk_registration.SetMetricSamplingStrategy(self.sitk_registration.RANDOM)
        self._verbose = verbose
        if verbose:
            self.sitk_registration.AddCommand(SimpleITK.sitkIterationEvent, self.print_state)

    def print_state(self):
        method = self.sitk_registration
        print(f"{method.GetOptimizerIteration():3} = {method.GetMetricValue():10.5f} : {method.GetOptimizerPosition()}")

    def register(self, fixed_image: SimpleITK.Image, moving_image: SimpleITK.Image):
        mask = (fixed_image != 0) & (moving_image != 0)
        self.sitk_registration.SetMetricFixedMask(mask)
        return self.sitk_registration.Execute(fixed_image, moving_image)

@main_wrapper
def main():
    reg = VertebraRegistration(verbose=True)
    for example in EXAMPLES:
        fixed = SimpleITK.ReadImage(example.fixed_path)
        moving = SimpleITK.ReadImage(example.moving_path)
        transformation = reg.register(fixed, moving)
        print(transformation)
        resampler = SimpleITK.ResampleImageFilter()
        resampler.SetReferenceImage(fixed)
        resampler.SetInterpolator(SimpleITK.sitkLinear)
        resampler.SetDefaultPixelValue(100)
        resampler.SetTransform(transformation)
        moved = resampler.Execute(moving)
        SimpleITK.WriteImage(moved, example.moved_path)
        print('Wrote', example.moved_path)


class RegistrationExample(EBC):
    def __init__(self, fixed_path: str, moving_path: str, moved_path: str):
        self.fixed_path = fixed_path
        self.moving_path = moving_path
        self.moved_path = moved_path


EXAMPLES = [
    RegistrationExample(r"C:\Users\Eren\Programme\Fractures\img\generated\patches\1001_L2_0_db2k_(60,50,40).nii.gz",
                        r"C:\Users\Eren\Programme\Fractures\img\generated\patches\1001_L1_0_db2k_(60,50,40).nii.gz",
                        r"C:\Users\Eren\Programme\Fractures\img\generated\patches\1001_L1L2_0_db2k_(60,50,40).nii.gz", )
]

if __name__ == '__main__':
    main()
