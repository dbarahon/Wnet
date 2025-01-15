program test_Wnet
  use Wnet !tested against python
  implicit none
  integer, parameter :: dp = selected_real_kind(15, 307)
  integer, parameter :: Ns = 10  ! Number of samples
  integer, parameter :: NF = 11  ! Number of samples
  real(dp) :: input(Ns, NF)
  real(dp) :: output(Ns)
  real(dp) :: perturbed_mean(NF)
  integer :: i, j
  real(dp) :: rand
  integer :: seed_size
  integer, allocatable :: seed(:)
  real(dp) :: mean(NF) = (/243.9, 0.6, 6.3, 0.013, 0.0002, 0.002, 9.75e-7, 7.87e-6, 889., 23.4, 1./)
  
  ! Initialize random seed for reproducibility
  call random_seed(size=seed_size)
  allocate(seed(seed_size))
  seed = 12345  ! Fixed seed for reproducibility
  call random_seed(put=seed)
  deallocate(seed)
  WRITE(*,"(14A)") ,  "   'T',    'AIRD',    'U',    'V',    'W',  'QV',    'QI',  'PBLH',    'TQV',    'AIRD_sfc'	"
  ! Generate 10 samples with perturbed means for input
  do i = 1, Ns
    do j = 1, NF
      call random_number(rand)      
      rand = 0.6_dp + (1.4_dp - 0.6_dp) * rand ! Scale perturbation ([0.6, 1.4] 
      !print *, rand    
      perturbed_mean(j) = mean(j)*rand
    end do

    write(*,"(14ES10.2)") perturbed_mean
    ! Assign perturbed mean to input sample
    input(i, :) = perturbed_mean
  end do

  call run_Wnet(input, output, Ns)

  print *, "SigmaW (m s-1):" 
  do i = 1, Ns
    print *, "Sample ", i, ": ", output(i)
  end do
end program test_Wnet
