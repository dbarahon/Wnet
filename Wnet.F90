module Wnet
  implicit none
  private
  integer, parameter :: dp = selected_real_kind(15, 307)
  integer, parameter :: NF = 14  ! Number of features
  integer, parameter :: NL = 5   ! Number of hidden layers
  integer, parameter :: NODES = 128

  real(dp), allocatable :: W_hidden(:, :, :) ! Shape: [NODES, NODES, NL]
  real(dp), allocatable :: b_hidden(:, :)    ! Shape: [NODES, NL]
  real(dp), allocatable :: W_input(:, :)     ! Shape: [NF, NODES]
  real(dp), allocatable :: b_input(:)        ! Shape: [NODES]
  real(dp), allocatable :: W_output(:)       ! Shape: [NODES]
  real(dp) :: b_output

  ! Module-level mean and standard deviation
  real(dp) :: mean(NF) = (/ &
      243.9, 0.6, 6.3, 0.013, 0.0002, 5.04, 21.8, 0.002, 9.75e-7, 7.87e-6, 0.6, 5.04, 21.8, 0.002/)
  real(dp) :: stddev(NF) = (/30.3, 0.42, 16.1, 7.9, 0.05, 20.6, 20.8, 0.0036, 7.09e-6, 2.7e-5, 0.42, 20.6, 20.8, 0.0036/)
  
  public :: run_Wnet    

!========== Parameterization of SigmaW for atmospheric models ====================
!This module calculates the subgrid scale standard deviation in vertical velocity using the Wnet model according to:
! Barahona, D., Breen, K. H., Kalesse-Los, H., & Röttenbacher, J. (2024). 
!Deep Learning Parameterization of Vertical Wind Velocity Variability via Constrained Adversarial Training. 
!Artificial Intelligence for the Earth Systems, 3(1), e230025. doi: 10.1175/AIES-D-23-0025.1
!
!Wnet is called using run_Wnet(input, output, Ns) where Ns is the number of samples.
!The input to Wnet consists of a 14-dimensional vector [Ns, 14] including the Richardson number (Ri, dimensionless), total scalar
!diffusivity for momentum (KM, in m2 s-1), the 3-dimensional wind velocity (U, V and
!W, in m s-1), the water vapor, liquid and ice mass mixing ratios (QV, QL and QI
!in Kg Kg-1), air density (AIRD, in Kg m-3), and temperature (T in K). For each sample they must be in the order:
!['T', 'AIRD', 'U', 'V', 'W', 'KM', 'RI', 'QV', 'QI', 'QL', AIRD_sfc, 'KM_sfc', 'RI_sfc', 'QV_sfc']  
!The subscript _sfc indicates surface values.
!
!Output (shape [Ns]) is the standard deviation in W (m s-1)
! 
!This version has been optimized for parallel performance (use the -fopenmp -O3 flags).
!
!Developed by Donifan Barahona donifan.o.barahona@nasa.gov

contains

!========================================
!weight loading
!========================================
subroutine load_weights()
    implicit none
    integer :: i, j, k
    character(len=10) :: header
    logical :: file_exists

    inquire(file='Wnet_weights.txt', exist=file_exists)
    if (.not. file_exists) then
        print *, "Error: File 'Wnet_weights.txt' not found!"
        stop
    end if

    open(10, file='Wnet_weights.txt', status='old')

    if (.not. allocated(W_input)) allocate(W_input(NF, NODES))
    if (.not. allocated(b_input)) allocate(b_input(NODES))
    if (.not. allocated(W_hidden)) allocate(W_hidden(NODES, NODES, NL))
    if (.not. allocated(b_hidden)) allocate(b_hidden(NODES, NL))
    if (.not. allocated(W_output)) allocate(W_output(NODES))

    read(10, *) header, i, j
    read(10, *) ((W_input(i, j), j = 1, NODES), i = 1, NF)
    read(10, *) header, i
    read(10, *) (b_input(i), i = 1, NODES)

    do k = 1, NL
        read(10, *) header, i, j
        read(10, *) ((W_hidden(i, j, k), j = 1, NODES), i = 1, NODES)
        read(10, *) header, i
        read(10, *) (b_hidden(i, k), i = 1, NODES)
    end do

    read(10, *) header, i
    read(10, *) (W_output(i), i = 1, NODES)
    read(10, *) header, i
    read(10, *) b_output

    close(10)
end subroutine load_weights

!========================================
! Leaky ReLU 
!========================================
function leaky_relu(x, alpha) result(y)
    real(dp), intent(in) :: x, alpha
    real(dp) :: y
    if (x < 0.0_dp) then
        y = alpha * x
    else
        y = x
    end if
end function leaky_relu

!========================================
! Standardize Input Data
!========================================
subroutine standardize_input(input, Ns)
    integer, intent(in) :: Ns
    real(dp), intent(inout) :: input(Ns, NF)

    integer :: i, j
    !$OMP PARALLEL DO PRIVATE(i, j) SCHEDULE(static)
    do i = 1, Ns
        do j = 1, NF
            if (stddev(j) /= 0.0_dp) then
                input(i, j) = (input(i, j) - mean(j)) / stddev(j)
            else
                input(i, j) = 0.0_dp
            end if
        end do
    end do
    !$OMP END PARALLEL DO
end subroutine standardize_input

!========================================
! Forward Pass
!========================================
subroutine forward_pass(input, output, Ns)
    integer, intent(in) :: Ns
    real(dp), intent(in) :: input(Ns, NF)
    real(dp), intent(out) :: output(Ns)

    integer :: i, j, k
    real(dp), allocatable :: layer_output(:,:), temp_output(:,:)
    real(dp) :: alpha

    allocate(layer_output(Ns, NODES))
    allocate(temp_output(Ns, NODES))

    alpha = 0.2_dp
    temp_output = matmul(input, W_input)

    !$OMP PARALLEL DO PRIVATE(i, j) SCHEDULE(static)
    do i = 1, Ns
        do j = 1, NODES
            layer_output(i, j) = leaky_relu(temp_output(i, j) + b_input(j), alpha)
        end do
    end do
    !$OMP END PARALLEL DO

    do k = 1, NL
        temp_output = matmul(layer_output, W_hidden(:, :, k))
        if (k == NL) alpha = 0.1_dp

        !$OMP PARALLEL DO PRIVATE(i, j) SCHEDULE(static)
        do i = 1, Ns
            do j = 1, NODES
                layer_output(i, j) = leaky_relu(temp_output(i, j) + b_hidden(j, k), alpha)
            end do
        end do
        !$OMP END PARALLEL DO
    end do

    !$OMP PARALLEL DO PRIVATE(i) SCHEDULE(static)
    do i = 1, Ns
        output(i) = dot_product(layer_output(i, :), W_output) + b_output
    end do
    !$OMP END PARALLEL DO

    deallocate(layer_output, temp_output)
end subroutine forward_pass

!!!!!!!!!!!!!!!!!
subroutine run_Wnet(input, output, Ns)
    integer, intent(in) :: Ns
    real(dp), intent(inout) :: input(Ns, NF)
    real(dp), intent(out) :: output(Ns)

    call load_weights()
    call standardize_input(input, Ns)
    call forward_pass(input, output, Ns)
end subroutine run_Wnet

end module Wnet
