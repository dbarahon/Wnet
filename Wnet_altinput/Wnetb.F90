module Wnet
  implicit none
  private
  integer, parameter :: dp = selected_real_kind(15, 307)
  integer, parameter :: NF = 11  ! Number of features in Wnetb
  integer, parameter :: NF1 = 14  ! Number of features in Wnet
  integer, parameter :: NL = 5   ! Number of hidden layers
  integer, parameter :: NODES = 128

  
  real(dp), allocatable :: W_input(:, :)     ! Shape: [NF, NODES]
  real(dp), allocatable :: b_input(:)        ! Shape: [NODES]
  real(dp), allocatable :: W_hidden_tr(:, :) ! Shape: [NODES, NF1]
  real(dp), allocatable :: b_hidden_tr(:)    ! Shape: [NF1]
  real(dp), allocatable :: W_input_b(:, :)     ! Shape: [NF1, NODES]
  real(dp), allocatable :: b_input_b(:)        ! Shape: [NODES] 
  real(dp), allocatable :: W_hidden(:, :, :) ! Shape: [NODES, NODES, NL]
  real(dp), allocatable :: b_hidden(:, :)    ! Shape: [NODES, NL]  
  real(dp), allocatable :: W_output(:)       ! Shape: [NODES]
  real(dp) :: b_output

  ! Module-level mean and standard deviation
  real(dp) :: mean(NF) = (/243.9, 0.6, 6.3, 0.013, 0.0002, 0.002, 9.75e-7, 7.87e-6, 889., 23.4, 1./)
  real(dp) :: stddev(NF) = (/30.3, 0.42, 16.1, 7.9, 0.05, 0.0036, 7.09e-6,  2.7e-5, 387., 15.6, 0.1/)
  
  public :: run_Wnet    

!========== Parameterization of SigmaW ====================
!This module calculates the subgrid scale standard deviation in vertical velocity using the Wnet model according to:
! Barahona, D., Breen, K. H., Kalesse-Los, H., & Röttenbacher, J. (2024). 
!Deep Learning Parameterization of Vertical Wind Velocity Variability via Constrained Adversarial Training. 
!Artificial Intelligence for the Earth Systems, 3(1), e230025. doi: 10.1175/AIES-D-23-0025.1

!This module is based on a pretrained neural network that uses Wnet as a foundation model for subgrid variability, 
!to change the input set (Barahonat et al. 2025)
!
!
!Wnet is called using run_Wnet(input, output, Ns) where Ns is the number of samples.
!
!The input to Wnet consists of a 11-dimensional vector [Ns, 11] including the 3-dimensional wind velocity (U, V and
!W, in m s-1), the water vapor, liquid and ice mass mixing ratios (QV, QL and QI
!in Kg Kg-1), air density (ÏAIRD, in Kg m-3), and temperature (T in K), the boundary layer height (PBLH, m) and 
!the column-integrated water vapor (TQV, Kg m-2).   

!For each sample they must be in the order:
!['T', 'AIRD', 'U', 'V', 'W', 'QV', 'QI', 'QL', 'PBLH' 'TQV' 'AIRD_sfc']  
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

    inquire(file='Wnet_weights_b.txt', exist=file_exists)
    if (.not. file_exists) then
        print *, "Error: File 'Wnet_weights_b.txt' not found!"
        stop
    end if
  

    open(10, file='Wnet_weights_b.txt', status='old')

    if (.not. allocated(W_input)) allocate(W_input(NF, NODES))
    if (.not. allocated(b_input)) allocate(b_input(NODES))
    if (.not. allocated(W_hidden_tr)) allocate(W_hidden_tr(NODES, NF1))
    if (.not. allocated(b_hidden_tr)) allocate(b_hidden_tr(NF1))
    if (.not. allocated(W_input_b)) allocate(W_input_b(NF1, NODES))
    if (.not. allocated(b_input_b)) allocate(b_input_b(NODES))
    if (.not. allocated(W_hidden)) allocate(W_hidden(NODES, NODES, NL))
    if (.not. allocated(b_hidden)) allocate(b_hidden(NODES, NL))
    if (.not. allocated(W_output)) allocate(W_output(NODES))

    !input layer 
    read(10, *) header, i, j
    read(10, *) ((W_input(i, j), j = 1, NODES), i = 1, NF)
    read(10, *) header, i
    read(10, *) (b_input(i), i = 1, NODES)
    
    !Translation layer 
    read(10, *) header, i, j
    read(10, *) ((W_hidden_tr(i, j), j = 1, NF1), i = 1, NODES)
    read(10, *) header, i
    read(10, *) (b_hidden_tr(i), i = 1, NF1)
    
    !original input layer 
    read(10, *) header, i, j
    read(10, *) ((W_input_b(i, j), j = 1, NODES), i = 1, NF1)
    read(10, *) header, i
    read(10, *) (b_input_b(i), i = 1, NODES)    

    !original hidden layer
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
    real(dp), allocatable :: layer_output(:,:), layer_output_tr(:,:), temp_output(:,:), temp_output_tr(:,:)
    real(dp) :: alpha

    allocate(layer_output(Ns, NODES))
    allocate(temp_output(Ns, NODES))
    allocate(layer_output_tr(Ns, NF1))
    allocate(temp_output_tr(Ns, NF1))

    alpha = 0.01_dp
    temp_output = matmul(input, W_input)

    !===input layer ========
    !$OMP PARALLEL DO PRIVATE(i, j) SCHEDULE(static)
    do i = 1, Ns
        do j = 1, NODES
            layer_output(i, j) = leaky_relu(temp_output(i, j) + b_input(j), alpha)
        end do
    end do
    !$OMP END PARALLEL DO
    
    !===transformation layer ========
    
    temp_output_tr = matmul(layer_output, W_hidden_tr(:, :))
    !$OMP PARALLEL DO PRIVATE(i, j) SCHEDULE(static)
    do i = 1, Ns
        do j = 1, NF1
            layer_output_tr(i, j) = leaky_relu(temp_output_tr(i, j) + b_hidden_tr(j), alpha)
        end do
    end do
    !$OMP END PARALLEL DO

    !=== original Wnet input ========
    
    alpha = 0.2_dp
    
    temp_output = matmul(layer_output_tr, W_input_b(:, :))
    !$OMP PARALLEL DO PRIVATE(i, j) SCHEDULE(static)
    do i = 1, Ns
        do j = 1, NODES
            layer_output(i, j) = leaky_relu(temp_output(i, j) + b_input_b(j), alpha)
        end do
    end do
    !$OMP END PARALLEL DO
        
    !=== original Wnet hidden layers ======== 
        
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

    !=== Wnet output layers ======== 
    
    !$OMP PARALLEL DO PRIVATE(i) SCHEDULE(static)
    do i = 1, Ns
        output(i) = dot_product(layer_output(i, :), W_output) + b_output
    end do
    !$OMP END PARALLEL DO

    deallocate(layer_output, temp_output, layer_output_tr, temp_output_tr)
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
