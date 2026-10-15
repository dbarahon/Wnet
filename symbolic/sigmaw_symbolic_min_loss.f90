module sigmaw_symbolic_mod
  implicit none
  private
  public :: compute_sigmaw_symbolic
  integer, parameter :: r8 = selected_real_kind(12, 300)

contains

  pure real(r8) function sqrt_abs(x)
    real(r8), intent(in) :: x
    sqrt_abs = sqrt(abs(x))
  end function sqrt_abs

  pure real(r8) function log_abs(x)
    real(r8), intent(in) :: x
    log_abs = log(abs(x) + 1.0e-6_r8)
  end function log_abs

  pure real(r8) function exp_clip(x)
    real(r8), intent(in) :: x
    exp_clip = exp(min(max(x, -20.0_r8), 20.0_r8))
  end function exp_clip

  pure real(r8) function square(x)
    real(r8), intent(in) :: x
    square = x*x
  end function square

  pure real(r8) function cube(x)
    real(r8), intent(in) :: x
    cube = x*x*x
  end function cube

  pure real(r8) function pressure_to_hpa(p_in)
    real(r8), intent(in) :: p_in
    if (abs(p_in) > 2000.0_r8) then
      pressure_to_hpa = 0.01_r8 * p_in
    else
      pressure_to_hpa = p_in
    end if
  end function pressure_to_hpa

  subroutine compute_sigmaw_symbolic(T, P, Z, AIRD, U, V, QV, PS, PBLH, TQV, AIRD_surf, sigmaw)
    ! Auto-generated from evaluate_merra2_wstd_expression_v10_zero_minfortran.py
    ! Equation selection: minimum_loss
    ! LHS: none
    ! RHS: (0.8146682 - (((((square(PS) - ((RHO_DIFF * WIND) - (abs(PBLH) + -0.5929808))) * (((P * square(TQV)) + sqrt_abs(AIRD_surf / 1.8222452)) * 0.32735726)) + (abs(AIRD_surf + PS) + (PS + abs((P * QV) + ((LOGP * QV) + (RHO_DIFF - T)))))) + (((log_abs(0.110092536 - (AIRD_surf / 1.8403813)) * (abs(RHO_DIFF) / -1.1952053)) + ((TQV - (((AIRD_surf + ABOVE_PBL) * 0.30626255) + 0.037839685)) + ((abs(ABOVE_PBL) + T) + ((WIND + AIRD_surf) * P)))) - (log_abs(abs(RHO_DIFF) + (((AIRD_surf + LOGP) + LOGP) * QV)) * T))) * -0.08306832)) * ((2.296377 - sqrt_abs(P + -0.46593115)) * 0.4346805)
    ! Target inverse: Wstd = max(expression, 0)^2
    ! RHS feature symbols used: T, P, LOGP, WIND, QV, PS, PBLH, TQV, AIRD_surf, RHO_DIFF, ABOVE_PBL
    ! Minimal raw inputs used: T, P, Z, AIRD, U, V, QV, PS, PBLH, TQV, AIRD_surf
    ! Units expected by this generated evaluator:
    !   T [K], P [hPa or Pa], Z/H [m], AIRD [kg m-3], U/V/W [m s-1],
    !   QV/QI/QL [kg kg-1], PS in the same raw units used by training,
    !   PBLH [m], TQV [kg m-2], AIRD_surf [kg m-3].
    real(r8), intent(in)  :: T(:,:,:)
    real(r8), intent(in)  :: P(:,:,:)
    real(r8), intent(in)  :: Z(:,:,:)
    real(r8), intent(in)  :: AIRD(:,:,:)
    real(r8), intent(in)  :: U(:,:,:)
    real(r8), intent(in)  :: V(:,:,:)
    real(r8), intent(in)  :: QV(:,:,:)
    real(r8), intent(in)  :: PS(:,:)
    real(r8), intent(in)  :: PBLH(:,:)
    real(r8), intent(in)  :: TQV(:,:)
    real(r8), intent(in)  :: AIRD_surf(:,:)
    real(r8), intent(out) :: sigmaw(:,:,:)
    integer :: i, j, k
    integer :: im, jm, lm
    real(r8) :: p_hpa
    real(r8) :: fit
    real(r8) :: tmp_sigma
    real(r8) :: v_T
    real(r8) :: v_P
    real(r8) :: v_LOGP
    real(r8) :: v_WIND
    real(r8) :: v_U
    real(r8) :: v_V
    real(r8) :: v_QV
    real(r8) :: v_PS
    real(r8) :: v_PBLH
    real(r8) :: v_TQV
    real(r8) :: v_AIRD_surf
    real(r8) :: v_RHO_DIFF
    real(r8) :: v_AIRD
    real(r8) :: v_ABOVE_PBL
    real(r8) :: v_H
    real(r8) :: x_T
    real(r8) :: x_P
    real(r8) :: x_LOGP
    real(r8) :: x_WIND
    real(r8) :: x_QV
    real(r8) :: x_PS
    real(r8) :: x_PBLH
    real(r8) :: x_TQV
    real(r8) :: x_AIRD_surf
    real(r8) :: x_RHO_DIFF
    real(r8) :: x_ABOVE_PBL

    im = size(sigmaw, 1)
    jm = size(sigmaw, 2)
    lm = size(sigmaw, 3)

    if (size(T,1) /= im .or. size(T,2) /= jm .or. size(T,3) /= lm) stop "T has incompatible shape"
    if (size(P,1) /= im .or. size(P,2) /= jm .or. size(P,3) /= lm) stop "P has incompatible shape"
    if (size(Z,1) /= im .or. size(Z,2) /= jm .or. size(Z,3) /= lm) stop "Z has incompatible shape"
    if (size(AIRD,1) /= im .or. size(AIRD,2) /= jm .or. size(AIRD,3) /= lm) stop "AIRD has incompatible shape"
    if (size(U,1) /= im .or. size(U,2) /= jm .or. size(U,3) /= lm) stop "U has incompatible shape"
    if (size(V,1) /= im .or. size(V,2) /= jm .or. size(V,3) /= lm) stop "V has incompatible shape"
    if (size(QV,1) /= im .or. size(QV,2) /= jm .or. size(QV,3) /= lm) stop "QV has incompatible shape"
    if (size(PS,1) /= im .or. size(PS,2) /= jm) stop "PS has incompatible shape"
    if (size(PBLH,1) /= im .or. size(PBLH,2) /= jm) stop "PBLH has incompatible shape"
    if (size(TQV,1) /= im .or. size(TQV,2) /= jm) stop "TQV has incompatible shape"
    if (size(AIRD_surf,1) /= im .or. size(AIRD_surf,2) /= jm) stop "AIRD_surf has incompatible shape"
    do k = 1, lm
      do j = 1, jm
        do i = 1, im
        v_T = T(i,j,k)
        v_P = P(i,j,k)
        v_H = Z(i,j,k)
        v_AIRD = AIRD(i,j,k)
        v_U = U(i,j,k)
        v_V = V(i,j,k)
        v_QV = QV(i,j,k)
        v_PS = PS(i,j)
        v_PBLH = PBLH(i,j)
        v_TQV = TQV(i,j)
        v_AIRD_surf = AIRD_surf(i,j)
        p_hpa = pressure_to_hpa(v_P)
        v_WIND = sqrt(v_U*v_U + v_V*v_V)
        v_RHO_DIFF = v_AIRD_surf - v_AIRD
        v_LOGP = log(max(p_hpa, 1.0e-6_r8))
        v_ABOVE_PBL = v_H - v_PBLH
        ! T <= T, mean=250.97157900100709, std=29.732862747893407
        x_T = (v_T - 2.50971579001007086e+02_r8) / 2.97328627478934067e+01_r8
        ! P <= P, mean=525.30462499999999, std=330.69534775773514
        x_P = (v_P - 5.25304624999999987e+02_r8) / 3.30695347757735135e+02_r8
        ! LOGP <= LOGP, mean=5.8452266579046253, std=1.1455057813227769
        x_LOGP = (v_LOGP - 5.84522665790462526e+00_r8) / 1.14550578132277692e+00_r8
        ! WIND <= WIND, mean=12.921231704571683, std=10.909896623785327
        x_WIND = (v_WIND - 1.29212317045716834e+01_r8) / 1.09098966237853272e+01_r8
        ! QV <= QV, mean=0.0026222415675365607, std=0.0040788054983921162
        x_QV = (v_QV - 2.62224156753656070e-03_r8) / 4.07880549839211622e-03_r8
        ! PS <= PS, mean=98054.56724982812, std=7705.5305271623256
        x_PS = (v_PS - 9.80545672498281201e+04_r8) / 7.70553052716232560e+03_r8
        ! PBLH <= PBLH, mean=778.28174161488721, std=595.13961199536459
        x_PBLH = (v_PBLH - 7.78281741614887210e+02_r8) / 5.95139611995364589e+02_r8
        ! TQV <= TQV, mean=20.301259514678666, std=16.847319930593645
        x_TQV = (v_TQV - 2.03012595146786659e+01_r8) / 1.68473199305936454e+01_r8
        ! AIRD_surf <= AIRD_surf, mean=1.2190244213421941, std=0.091519331705151696
        x_AIRD_surf = (v_AIRD_surf - 1.21902442134219413e+00_r8) / 9.15193317051516964e-02_r8
        ! RHO_DIFF <= RHO_DIFF, mean=0.53021350774842502, std=0.39648808178840761
        x_RHO_DIFF = (v_RHO_DIFF - 5.30213507748425017e-01_r8) / 3.96488081788407609e-01_r8
        ! ABOVE_PBL <= ABOVE_PBL, mean=6877.7428943775367, std=7686.4881564799016
        x_ABOVE_PBL = (v_ABOVE_PBL - 6.87774289437753669e+03_r8) / 7.68648815647990159e+03_r8
        fit = (0.9981903025485e0_r8 - 0.4346805e0_r8*sqrt_abs(x_P &
          & - 0.46593115000000002))*( & -0.025440715507416e0_r8*x_ABOVE_PBL - 0.025440715507416e0_r8* & x_AIRD_surf &
          & + 0.08306832e0_r8*x_P*(x_AIRD_surf + x_WIND) + & 0.08306832e0_r8*x_PS &
          & - 0.08306832e0_r8*x_T*log_abs(x_QV*(x_AIRD_surf + & 2.0*x_LOGP) + Abs(x_RHO_DIFF)) &
          & + 0.08306832e0_r8*x_T + 0.08306832e0_r8* & x_TQV + 0.08306832e0_r8*(0.32735726e0_r8*x_P*square(x_TQV) &
          & + & 0.32735726e0_r8*sqrt_abs(0.54877356790403398*x_AIRD_surf))*( & -x_RHO_DIFF*x_WIND + square(x_PS) &
          & + abs(x_PBLH) - 0.5929808e0_r8) - & 0.0695012982288482e0_r8*log_abs(0.110092536 &
          & - 0.54336566014879639* & x_AIRD_surf)*abs(x_RHO_DIFF) + 0.08306832e0_r8*abs(x_ABOVE_PBL) &
          & + & 0.08306832e0_r8*abs(x_AIRD_surf + x_PS) + 0.08306832e0_r8*abs(x_LOGP* & x_QV + x_P*x_QV &
          & + x_RHO_DIFF - x_T) + 0.811524920937721e0_r8)
        tmp_sigma = max(fit, 0.0_r8)**2
        sigmaw(i,j,k) = max(tmp_sigma, 0.00000000000000000e+00_r8)
        end do
      end do
    end do
  end subroutine compute_sigmaw_symbolic

end module sigmaw_symbolic_mod
